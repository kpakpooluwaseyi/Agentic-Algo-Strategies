
import json
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.signal import find_peaks
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def preprocess_data(df: pd.DataFrame, **params):
    """
    Applies indicator calculations and pattern detection to the input DataFrame.
    """
    # Sanitize column names (e.g., ' open' -> 'Open')
    df.columns = [col.strip().title() for col in df.columns]

    # Parameters from the strategy class
    fast_ema_period = params.get('fast_ema_period', 50)
    slow_ema_period = params.get('slow_ema_period', 200)
    atr_period = params.get('atr_period', 14)

    # -- Indicator Calculations --
    # --- Multi-Timeframe Trend Filter (4H) ---
    # Resample to 4-hour timeframe
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Calculate a 12-period EMA on the 4H data (roughly equivalent to 200-period on 15m)
    df_4h.ta.ema(length=slow_ema_period // 16, append=True, col_names=(f'EMA_4H',))

    # Determine the 4H trend
    df_4h['trend_4h'] = np.where(df_4h['Close'] > df_4h['EMA_4H'], 1, -1)

    # Map the 4H trend back to the 15-minute dataframe
    df['trend_4h'] = df_4h['trend_4h'].reindex(df.index, method='ffill')

    # --- 15-Minute Indicators ---
    # Exponential Moving Averages for trend (now only used for visualization/confirmation if needed, slow EMA is replaced by 4H trend)
    df.ta.ema(length=slow_ema_period, append=True, col_names=(f'EMA_{slow_ema_period}',))

    # Average True Range for volatility (risk management)
    df.ta.atr(length=atr_period, append=True, col_names=(f'ATR_{atr_period}',))

    # Volume Moving Average for confirmation
    df.ta.sma(close=df['Volume'], length=20, append=True, col_names=('Volume_SMA_20',))

    # -- Swing Point Detection --
    peak_distance = params.get('peak_distance', 15)

    # Find swing highs
    high_peaks_indices, _ = find_peaks(df['High'], distance=peak_distance)
    df['swing_high'] = False
    if len(high_peaks_indices) > 0:
        df.iloc[high_peaks_indices, df.columns.get_loc('swing_high')] = True

    # Find swing lows
    low_peaks_indices, _ = find_peaks(-df['Low'], distance=peak_distance)
    df['swing_low'] = False
    if len(low_peaks_indices) > 0:
        df.iloc[low_peaks_indices, df.columns.get_loc('swing_low')] = True

    # -- Candlestick Pattern Detection --
    candle_range = df['High'] - df['Low']
    body_size = abs(df['Close'] - df['Open'])

    # Doji: Small body, can be bullish or bearish signal depending on context.
    is_doji = body_size < (candle_range * 0.1)

    # Hammer (Bullish): Small body at the top, long lower wick.
    is_hammer = (body_size < (candle_range * 0.3)) & \
                ((df['Close'] - df['Low']) > (candle_range * 0.6)) & \
                ((df['High'] - df['Close']) < (candle_range * 0.2))

    # Shooting Star (Bearish): Small body at the bottom, long upper wick.
    is_shooting_star = (body_size < (candle_range * 0.3)) & \
                       ((df['High'] - df['Open']) > (candle_range * 0.6)) & \
                       ((df['Open'] - df['Low']) < (candle_range * 0.2))

    # Bullish Engulfing
    is_prev_bearish = df['Close'].shift(1) < df['Open'].shift(1)
    is_curr_bullish = df['Close'] > df['Open']
    body_engulfs = (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))
    is_bullish_engulfing = is_prev_bearish & is_curr_bullish & body_engulfs

    # Bearish Engulfing
    is_prev_bullish = df['Close'].shift(1) > df['Open'].shift(1)
    is_curr_bearish = df['Close'] < df['Open']
    body_engulfs_bearish = (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))
    is_bearish_engulfing = is_prev_bullish & is_curr_bearish & body_engulfs_bearish

    # --- Composite Reversal Signals ---
    df['bullish_reversal'] = is_bullish_engulfing | is_hammer | is_doji
    df['bearish_reversal'] = is_bearish_engulfing | is_shooting_star | is_doji

    # Clean up NaN values resulting from indicator calculations
    df.dropna(inplace=True)

    # The backtesting framework requires specific column names. Rename if necessary.
    # pandas_ta might create names like 'EMA_50'. The framework will find them.
    # Let's ensure the core OHLCV are there.
    df.rename(columns={
        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
    }, inplace=True)

    return df

class ConservativeTrendlineReversal(Strategy):
    """
    Strategy that trades reversals at a trendline proxy (EMA) in the direction of a larger trend.
    """

    # === Strategy Parameters ===
    slow_ema_period = 200
    atr_period = 14
    atr_multiplier_sl = 2
    atr_multiplier_tp = 6  # Adjusted for 1:3 Risk/Reward ratio
    peak_distance = 15
    entry_buffer_pct = 0.001  # 0.1% buffer for pending order

    def init(self):
        """
        Initialize indicators and state variables.
        """
        # Indicators
        self.slow_ema = self.I(lambda x: x, self.data.df[f'EMA_{self.slow_ema_period}'])
        self.atr = self.I(lambda x: x, self.data.df[f'ATR_{self.atr_period}'])
        self.volume_sma = self.I(lambda x: x, self.data.df['Volume_SMA_20'])
        self.bullish_reversal = self.I(lambda x: x, self.data.df['bullish_reversal'])
        self.bearish_reversal = self.I(lambda x: x, self.data.df['bearish_reversal'])
        self.swing_high = self.I(lambda x: x, self.data.df['swing_high'])
        self.swing_low = self.I(lambda x: x, self.data.df['swing_low'])
        self.trend_4h = self.I(lambda x: x, self.data.df['trend_4h'])

        # State machine for pending orders
        self.pending_order_direction = 0  # 0 = None, 1 = Long, -1 = Short
        self.pending_entry_price = 0
        self.pending_sl = 0
        self.pending_tp = 0
        self.invalidation_price = 0

    def _get_trendline_value(self, swing_points_series, price_series):
        """
        Calculates the current value of a trendline based on the last two swing points.
        Returns the trendline value or None if a trendline can't be formed.
        """
        # Get indices of the last two swing points
        swing_indices = np.where(swing_points_series == 1)[0]
        if len(swing_indices) < 2:
            return None

        p2_idx, p1_idx = swing_indices[-2:]

        # Get price values at those points
        p1_price = price_series[p1_idx]
        p2_price = price_series[p2_idx]

        # Calculate the slope of the trendline
        # Using index difference for time component
        slope = (p1_price - p2_price) / (p1_idx - p2_idx)

        # Extrapolate to the current bar
        current_idx = len(price_series) - 1
        trendline_value = p1_price + slope * (current_idx - p1_idx)

        return trendline_value

    def next(self):
        """
        Define the trading logic using a state machine for pending orders.
        """
        price = self.data.Close[-1]

        # --- Stage 1: Manage open or pending orders ---
        if self.position:
            return  # Active trade management would go here (e.g., trailing SL)

        # If a pending order is active, check for trigger or invalidation
        if self.pending_order_direction != 0:
            # Check for entry trigger
            if self.pending_order_direction == 1 and self.data.High[-1] > self.pending_entry_price:
                self.buy(sl=self.pending_sl, tp=self.pending_tp)
                self.pending_order_direction = 0 # Reset state
            elif self.pending_order_direction == -1 and self.data.Low[-1] < self.pending_entry_price:
                self.sell(sl=self.pending_sl, tp=self.pending_tp)
                self.pending_order_direction = 0 # Reset state

            # Check for invalidation (e.g., price moves too far against the setup)
            elif (self.pending_order_direction == 1 and price < self.invalidation_price) or \
                 (self.pending_order_direction == -1 and price > self.invalidation_price):
                self.pending_order_direction = 0 # Cancel pending order

            return # Don't look for new signals if managing a pending order

        # --- Stage 2: Look for new trade setups ---
        # === Long Setup Conditions ===
        uptrend_line_val = self._get_trendline_value(self.swing_low, self.data.Low)
        if uptrend_line_val is not None:
            if (self.trend_4h[-1] == 1 and
                self.data.Low[-1] <= uptrend_line_val and
                self.bullish_reversal[-1] and
                self.data.Volume[-1] > self.volume_sma[-1]):

                # Setup a pending BUY STOP order
                self.pending_order_direction = 1
                self.pending_entry_price = self.data.High[-1] * (1 + self.entry_buffer_pct)
                self.pending_sl = self.data.Low[-1] - self.atr[-1] * self.atr_multiplier_sl
                self.pending_tp = self.pending_entry_price + (self.pending_entry_price - self.pending_sl) * (self.atr_multiplier_tp / self.atr_multiplier_sl)
                self.invalidation_price = self.data.Low[-1] # Invalidate if price drops below the signal candle's low

        # === Short Setup Conditions ===
        downtrend_line_val = self._get_trendline_value(self.swing_high, self.data.High)
        if downtrend_line_val is not None:
            if (self.trend_4h[-1] == -1 and
                  self.data.High[-1] >= downtrend_line_val and
                  self.bearish_reversal[-1] and
                  self.data.Volume[-1] > self.volume_sma[-1]):

                # Setup a pending SELL STOP order
                self.pending_order_direction = -1
                self.pending_entry_price = self.data.Low[-1] * (1 - self.entry_buffer_pct)
                self.pending_sl = self.data.High[-1] + self.atr[-1] * self.atr_multiplier_sl
                self.pending_tp = self.pending_entry_price - (self.pending_sl - self.pending_entry_price) * (self.atr_multiplier_tp / self.atr_multiplier_sl)
                self.invalidation_price = self.data.High[-1] # Invalidate if price rises above the signal candle's high

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to be JSON serializable,
    handling various data types that may not be directly convertible.
    """
    sanitized = {}
    for key, value in stats.items():
        # First, filter out DataFrame/Series objects which cause ambiguity
        if isinstance(value, (pd.DataFrame, pd.Series)):
            continue
        # Now, handle other types
        if pd.isna(value):
            sanitized[key] = None
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = value.item()
        elif isinstance(value, (int, float, str, bool)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized

if __name__ == '__main__':
    try:
        # Robustly load the data, ignoring the potentially malformed header
        col_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        data = pd.read_csv(
            'data/BTC-USD-15m.csv',
            header=0,
            names=col_names,
            index_col='datetime',
            parse_dates=True,
            usecols=range(len(col_names)) # Ensure only the named columns are read
        )
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure you have the correct data file.")
        # As a fallback, you could generate synthetic data here for testing purposes.
        exit()

    # Preprocess the data, passing strategy params
    strategy_params = {
        'slow_ema_period': ConservativeTrendlineReversal.slow_ema_period,
        'atr_period': ConservativeTrendlineReversal.atr_period,
        'peak_distance': ConservativeTrendlineReversal.peak_distance,
    }
    processed_data = preprocess_data(data.copy(), **strategy_params)

    # Instantiate the backtest
    bt = Backtest(
        processed_data,
        ConservativeTrendlineReversal,
        cash=100_000,
        commission=.002,
        finalize_trades=True
    )

    # Run the backtest
    stats = bt.run()
    print(stats)

    # Save the results
    results_path = 'results/temp_result.json'
    sanitized = sanitize_stats(stats)
    with open(results_path, 'w') as f:
        json.dump(sanitized, f, indent=4)

    print(f"Backtest stats saved to {results_path}")

    # Generate and save the plot
    plot_path = 'results/_conservative_trendline_reversal_.html'
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Backtest plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
