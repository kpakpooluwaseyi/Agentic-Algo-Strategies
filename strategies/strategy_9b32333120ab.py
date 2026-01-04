
from backtesting import Strategy
from backtesting.lib import crossover
import numpy as np
import pandas as pd
import talib
from scipy.signal import find_peaks

def preprocess_data(df: pd.DataFrame, **params):
    """
    Adds necessary indicators and signals to the DataFrame for the Trendline Pullback strategy.
    """
    # -- Multi-Timeframe Trend Filter (4H EMA 200) --
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_uptrend'] = (df_4h['Close'] > df_4h['ema_200']).astype(int)

    # Map the 4H trend back to the 15m DataFrame
    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill').fillna(0)

    # -- ATR for Risk Management --
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # -- Volume Confirmation --
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # -- Fair Value Gap (FVG) Detection --
    # A bullish FVG is formed when the low of the 3rd candle is above the high of the 1st candle.
    # A bearish FVG is formed when the high of the 3rd candle is below the low of the 1st candle.
    df['fvg_bullish'] = (df['Low'].shift(-1) > df['High'].shift(1))
    df['fvg_bearish'] = (df['High'].shift(-1) < df['Low'].shift(1))

    # -- Order Block (OB) Detection (Simplified) --
    # A bullish OB is the last down-candle before a strong up-move.
    # A bearish OB is the last up-candle before a strong down-move.
    # This simplified version marks candles that could be OBs.
    is_up_move = df['Close'] > df['Open']
    is_down_move = df['Close'] < df['Open']

    # Potential Bullish OB: A down candle followed by a strong up candle
    df['ob_bullish'] = is_down_move.shift(1) & is_up_move & (df['Close'] > df['High'].shift(1))
    # Potential Bearish OB: An up candle followed by a strong down candle
    df['ob_bearish'] = is_up_move.shift(1) & is_down_move & (df['Close'] < df['Low'].shift(1))

    # -- Swing Points for Trendline Proxy --
    # Find peaks (swing highs) and troughs (swing lows)
    # The 'distance' parameter is crucial to avoid noise. It's the minimum bars between peaks.
    # Let's make it optimizable later, but start with a reasonable default.
    peak_indices, _ = find_peaks(df['High'], distance=params.get('peak_distance', 10))
    trough_indices, _ = find_peaks(-df['Low'], distance=params.get('peak_distance', 10))
    df['swing_high'] = False
    df['swing_low'] = False
    df.iloc[peak_indices, df.columns.get_loc('swing_high')] = True
    df.iloc[trough_indices, df.columns.get_loc('swing_low')] = True

    return df

class TrendlinePullback(Strategy):
    # -- Optimizable Parameters --
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 4.0  # Targeting a minimum of 1:4 R:R
    volume_factor = 1.2  # Volume must be 20% above its 20-period MA
    peak_distance = 10 # Parameter for swing point detection

    # -- State Variables --
    swing_highs = []
    swing_lows = []
    trendline_broken_up = False
    trendline_broken_down = False
    breakout_level = None

    def init(self):
        # -- Indicators --
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend, name="htf_uptrend")
        self.atr = self.I(lambda: self.data.atr, name="atr")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="volume_ma")

        # -- Signals --
        self.fvg_bullish = self.I(lambda: self.data.fvg_bullish, name="fvg_bullish")
        self.fvg_bearish = self.I(lambda: self.data.fvg_bearish, name="fvg_bearish")
        self.ob_bullish = self.I(lambda: self.data.ob_bullish, name="ob_bullish")
        self.ob_bearish = self.I(lambda: self.data.ob_bearish, name="ob_bearish")
        self.swing_high = self.I(lambda: self.data.swing_high, name="swing_high")
        self.swing_low = self.I(lambda: self.data.swing_low, name="swing_low")

    def next(self):
        # -- Data availability check --
        if pd.isna(self.atr[-1]) or pd.isna(self.volume_ma[-1]) or pd.isna(self.htf_uptrend[-1]):
            return

        current_price = self.data.Close[-1]

        # -- State Updates: Track last two swing points --
        if self.swing_high[-1]:
            self.swing_highs.append((len(self.data) - 1, self.data.High[-1]))
            if len(self.swing_highs) > 2:
                self.swing_highs.pop(0)

        if self.swing_low[-1]:
            self.swing_lows.append((len(self.data) - 1, self.data.Low[-1]))
            if len(self.swing_lows) > 2:
                self.swing_lows.pop(0)

        # -- Filters --
        if self.position:
            return

        # -- Trendline Break Logic --
        # Check for established downtrend (lower highs) and a breakout
        if len(self.swing_highs) == 2 and not self.trendline_broken_up:
            if self.swing_highs[1][1] < self.swing_highs[0][1]: # Lower high
                if current_price > self.swing_highs[1][1]: # Breakout
                    self.trendline_broken_up = True
                    self.trendline_broken_down = False
                    self.breakout_level = self.swing_highs[1][1]

        # Check for established uptrend (higher lows) and a breakdown
        if len(self.swing_lows) == 2 and not self.trendline_broken_down:
            if self.swing_lows[1][1] > self.swing_lows[0][1]: # Higher low
                if current_price < self.swing_lows[1][1]: # Breakdown
                    self.trendline_broken_down = True
                    self.trendline_broken_up = False
                    self.breakout_level = self.swing_lows[1][1]

        # -- State Invalidation Logic --
        # If a breakout occurred but no entry, and price makes a new high/low, invalidate the signal
        if self.trendline_broken_up and current_price > self.breakout_level + self.atr[-1]:
             self.trendline_broken_up = False

        if self.trendline_broken_down and current_price < self.breakout_level - self.atr[-1]:
            self.trendline_broken_down = False

        # -- Entry Logic --

        # Long Entry: HTF is uptrend, trendline broken up, now waiting for pullback FVG/OB
        if self.htf_uptrend[-1] and self.trendline_broken_up:
            is_pullback = current_price < self.breakout_level

            if is_pullback and (self.fvg_bullish[-1] or self.ob_bullish[-1]):
                # Volume Confirmation
                if self.data.Volume[-1] > self.volume_ma[-1] * self.volume_factor:
                    sl = current_price - (self.atr[-1] * self.atr_sl_multiplier)
                    tp = current_price + (self.atr[-1] * self.atr_tp_multiplier)

                    if tp > current_price and sl < current_price: # Basic validation
                        self.buy(sl=sl, tp=tp)
                        self.trendline_broken_up = False # Reset state

        # Short Entry: HTF is downtrend, trendline broken down, now waiting for pullback FVG/OB
        if not self.htf_uptrend[-1] and self.trendline_broken_down:
            is_pullback = current_price > self.breakout_level

            if is_pullback and (self.fvg_bearish[-1] or self.ob_bearish[-1]):
                 # Volume Confirmation
                if self.data.Volume[-1] > self.volume_ma[-1] * self.volume_factor:
                    sl = current_price + (self.atr[-1] * self.atr_sl_multiplier)
                    tp = current_price - (self.atr[-1] * self.atr_tp_multiplier)

                    if tp < current_price and sl > current_price: # Basic validation
                        self.sell(sl=sl, tp=tp)
                        self.trendline_broken_down = False # Reset state

if __name__ == '__main__':
    from backtesting import Backtest
    import json

    # -- Data Loading --
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        # Sanitize column names (e.g., ' open' -> 'Open')
        df.columns = [col.strip().capitalize() for col in df.columns]
    except FileNotFoundError:
        print("Data file not found. A sample DataFrame will be generated.")
        # Create a sample DataFrame for demonstration if the file is not found
        rng = pd.date_range('2020-01-01', periods=2000, freq='15min')
        df = pd.DataFrame(np.random.randn(2000, 5),
                          columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                          index=rng)
        df['Open'] = 20000 + df['Open'].cumsum()
        df['High'] = df['Open'] + abs(df['High'])
        df['Low'] = df['Open'] - abs(df['Low'])
        df['Close'] = df['Open'] + df['Close']
        df['Volume'] = abs(df['Volume']) * 100

    # -- Preprocessing --
    df = preprocess_data(df)
    # -- Backtesting --
    bt = Backtest(df, TrendlinePullback, cash=100_000, commission=.002)
    stats = bt.run()

    print("--- Trendline Pullback FVG/OB Strategy ---")
    print(stats)

    # -- Plotting --
    try:
        bt.plot(filename='results/strategy_9b32333120ab.html', open_browser=False)
        print("\nBacktest plot saved to 'results/strategy_9b32333120ab.html'")
    except Exception as e:
        print(f"\nCould not generate plot. Error: {e}")

    # -- Save Results --
    # Sanitize stats object for JSON serialization
    stats['_strategy'] = str(stats['_strategy']) # Convert strategy object to string
    sanitized_stats = {key: str(value) if isinstance(value, (pd.Timestamp, pd.Timedelta)) else value
                       for key, value in stats.items() if not isinstance(value, (pd.Series, pd.DataFrame))}

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)
