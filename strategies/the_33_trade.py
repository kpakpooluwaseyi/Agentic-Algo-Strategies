from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os
from scipy.signal import find_peaks
import pandas_ta as ta
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all required indicators to the DataFrame.
    """
    df.ta.ema(length=5, append=True)
    df.ta.ema(length=13, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.pivots(append=True)

    # Add VuManchu Cipher B
    df = cipher_b(df)

    # Add Volume MA
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

    # Calculate ADR
    df_daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    df_daily['true_range'] = ta.true_range(df_daily['High'], df_daily['Low'], df_daily['Close'])
    df_daily['ADR'] = df_daily['true_range'].rolling(window=14).mean()
    df['ADR'] = df.index.normalize().map(df_daily['ADR'])
    df['ADR'].ffill(inplace=True)

    return df

def find_swing_points(data, prominence=2, width=1):
    """
    Identifies swing highs and lows in the data.
    Returns an array with 1 for swing highs, -1 for swing lows, and 0 otherwise.
    """
    peaks, _ = find_peaks(data, prominence=prominence, width=width)
    troughs, _ = find_peaks(-data, prominence=prominence, width=width)

    swings = np.zeros(len(data))
    swings[peaks] = 1
    swings[troughs] = -1
    return swings

def deduplicate_swings(swings):
    """
    Removes consecutive duplicate swing signals, keeping only the first occurrence.
    """
    if len(swings) == 0:
        return swings

    deduplicated = np.copy(swings)
    for i in range(1, len(deduplicated)):
        if deduplicated[i] != 0 and deduplicated[i] == deduplicated[i-1]:
            deduplicated[i] = 0
    return deduplicated

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it's JSON-serializable.
    Converts specific numpy types and pandas objects to native Python types.
    """
    sanitized = {}
    for key, value in stats.items():
        if key == '_strategy':
            continue
        if isinstance(value, (pd.DataFrame, pd.Series)):
            sanitized[key] = None
        elif isinstance(value, (np.integer, np.int_)):
            sanitized[key] = int(value)
        elif isinstance(value, np.floating):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized


class The33Trade(Strategy):
    """
    Strategy based on the "3-3 trade" concept of multi-day and intraday cycle peaks.

    Note: Inherits from backtesting.Strategy for compatibility with the backtesting library,
    as requested implementation details align with this framework over the alternative
    MoonDevStrategy base class.
    """
    # Optimizable parameters
    sl_atr_multiplier = 2.0
    tp_atr_multiplier = 3.0
    prominence = 2
    width = 1
    day_lookback = 288 # 3 days of 15m candles
    intraday_lookback = 24 # 6 hours of 15m candles
    rsi_ob = 70
    rsi_os = 30

    def init(self):
        """
        Initialize indicators and strategy variables.
        """
        # Indicators from preprocess_data
        self.ema5 = self.I(lambda: self.data.df['EMA_5'])
        self.ema13 = self.I(lambda: self.data.df['EMA_13'])
        self.ema50 = self.I(lambda: self.data.df['EMA_50'])
        self.ema200 = self.I(lambda: self.data.df['EMA_200'])
        self.rsi = self.I(lambda: self.data.df['RSI_14'])
        self.atr = self.I(lambda: self.data.df['ATRr_14'])
        self.adr = self.I(lambda: self.data.df['ADR'])
        self.volume_ma = self.I(lambda: self.data.df['Volume_MA'])

        # Cipher B Indicators
        self.cipher_buy = self.I(lambda: self.data.df['buy_signal'])
        self.cipher_sell = self.I(lambda: self.data.df['sell_signal'])

        # Intraday (15m) swing points for cycle counting
        self.swing_points_15m = self.I(find_swing_points, self.data.Close, prominence=self.prominence, width=self.width)

        # Multi-day (1h) swing points
        df_1h = self.data.df.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()

        swing_points_1h_series = pd.Series(find_swing_points(df_1h['Close'].values, prominence=self.prominence, width=self.width), index=df_1h.index)

        # Map 1h swing points back to 15m timeframe
        self.data.df['swing_points_1h'] = self.data.df.index.floor('H').map(swing_points_1h_series)
        self.data.df['swing_points_1h'].fillna(0, inplace=True)
        self.swing_points_1h = self.I(lambda: self.data.df['swing_points_1h'].values)

        # 4H trend filter
        df_4h = self.data.df.resample('4H').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
        df_4h['EMA_50_4H'] = ta.ema(df_4h['Close'], length=50)
        self.data.df['EMA_50_4H'] = self.data.df.index.floor('4H').map(df_4h['EMA_50_4H'])
        self.data.df['EMA_50_4H'].ffill(inplace=True)
        self.ema50_4h = self.I(lambda: self.data.df['EMA_50_4H'].values)


    def next(self):
        """
        Defines the trading logic for each bar.
        """
        if len(self.swing_points_1h) < self.day_lookback or len(self.swing_points_15m) < self.intraday_lookback:
            return

        # --- Cycle Counts ---
        # Multi-day (3-day lookback on 1H swings)
        day_swings = self.swing_points_1h[-self.day_lookback:]
        deduplicated_day_swings = deduplicate_swings(day_swings)
        three_level_rise = np.sum(deduplicated_day_swings == 1) >= 3
        three_level_drop = np.sum(deduplicated_day_swings == -1) >= 3

        # Intraday (lookback on 15m swings)
        intraday_swings = self.swing_points_15m[-self.intraday_lookback:]
        three_level_rise_intra = np.sum(intraday_swings == 1) >= 3
        three_level_drop_intra = np.sum(intraday_swings == -1) >= 3

        # --- Reversal Patterns (simple check for M/W formations on 15m swings) ---
        swing_points_15m_array = np.asarray(self.swing_points_15m)
        is_m_formation = np.array_equal(swing_points_15m_array[-5:], [1, -1, 1, -1, 0]) or np.array_equal(swing_points_15m_array[-4:], [1, -1, 1, 0])
        is_w_formation = np.array_equal(swing_points_15m_array[-5:], [-1, 1, -1, 1, 0]) or np.array_equal(swing_points_15m_array[-4:], [-1, 1, -1, 0])

        # Pin Bar detection
        is_bullish_pin_bar = self.is_pin_bar(self.data, -1, is_bullish=True)
        is_bearish_pin_bar = self.is_pin_bar(self.data, -1, is_bullish=False)

        reversal_pattern_long = is_w_formation or is_bullish_pin_bar
        reversal_pattern_short = is_m_formation or is_bearish_pin_bar

        # --- Trend Filter ---
        is_uptrend = self.data.Close[-1] > self.ema50_4h[-1]
        is_downtrend = self.data.Close[-1] < self.ema50_4h[-1]

        # --- Entry Conditions ---
        if not self.position:
            # Short Entry (Counter-trend: look for rise in uptrend)
            if is_uptrend and three_level_rise and three_level_rise_intra and reversal_pattern_short and self.cipher_sell[-1] and self.rsi[-1] > self.rsi_ob and self.data.Volume[-1] > self.volume_ma[-1]:
                entry_price = self.data.Close[-1]
                stop_loss = entry_price + self.atr[-1] * self.sl_atr_multiplier
                take_profit = entry_price - self.atr[-1] * self.tp_atr_multiplier

                # ADR Check for plausible target
                if (entry_price - take_profit) < self.adr[-1]:
                    if entry_price < stop_loss:
                        self.sell(sl=stop_loss, tp=take_profit)

            # Long Entry (Counter-trend: look for drop in downtrend)
            elif is_downtrend and three_level_drop and three_level_drop_intra and reversal_pattern_long and self.cipher_buy[-1] and self.rsi[-1] < self.rsi_os and self.data.Volume[-1] > self.volume_ma[-1]:
                entry_price = self.data.Close[-1]
                stop_loss = entry_price - self.atr[-1] * self.sl_atr_multiplier
                take_profit = entry_price + self.atr[-1] * self.tp_atr_multiplier

                # ADR Check for plausible target
                if (take_profit - entry_price) < self.adr[-1]:
                    if entry_price > stop_loss:
                        self.buy(sl=stop_loss, tp=take_profit)

    def is_pin_bar(self, data, index, is_bullish, body_ratio_threshold=0.33, wick_ratio_threshold=0.5):
        """
        Checks if the candle at the given index is a pin bar.
        """
        open_price = data.Open[index]
        high_price = data.High[index]
        low_price = data.Low[index]
        close_price = data.Close[index]

        body_size = abs(close_price - open_price)
        total_range = high_price - low_price

        if total_range == 0:
            return False

        body_to_range_ratio = body_size / total_range

        if body_to_range_ratio > body_ratio_threshold:
            return False

        if is_bullish: # Hammer
            lower_wick_size = min(open_price, close_price) - low_price
            return lower_wick_size / total_range >= wick_ratio_threshold
        else: # Shooting Star
            upper_wick_size = high_price - max(open_price, close_price)
            return upper_wick_size / total_range >= wick_ratio_threshold

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Clean up column names
        data.columns = [c.strip().title() for c in data.columns]
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        data = preprocess_data(data)
    else:
        raise FileNotFoundError(f"Data file not found at {data_path}")

    bt = Backtest(data, The33Trade, cash=100000, commission=.002)

    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    sanitized_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    try:
        plot_filename = 'results/the_33_trade.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
