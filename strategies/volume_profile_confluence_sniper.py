
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolumeProfileConfluenceSniper(Strategy):
    """
    A strategy that looks for pullbacks to a confluence of support/resistance,
    previous day's POC (approximated by VWAP), and Fibonacci levels,
    then waits for a candlestick confirmation before entering.
    """
    # --- Strategy Parameters ---
    ema_period = 200
    fib_levels = [0.5, 0.618]
    confluence_prox_pct = 0.01  # Proximity percent to consider a confluence zone (1%)
    lookback_period = 96  # Bars to look back for recent swing high/low (1 day)
    sl_buffer_pct = 0.005  # Buffer for stop loss (0.5%)
    min_rr = 2.0  # Minimum required risk-to-reward ratio

    def init(self):
        """Initialize indicators and state variables."""
        self.ema = self.I(lambda x: pd.Series(x).ewm(span=self.ema_period, adjust=False).mean(), self.data.Close)

        # Map pre-calculated data
        self.prev_day_poc = self.data.prev_day_poc
        self.swing_high_sr = self.data.swing_high_sr
        self.swing_low_sr = self.data.swing_low_sr
        self.hvn_from_swing_high = self.data.hvn_from_swing_high
        self.hvn_from_swing_low = self.data.hvn_from_swing_low

    def _is_bullish_confirmation(self):
        """Checks for a bullish engulfing or pin bar pattern."""
        if len(self.data) < 2: return False

        # Bullish Engulfing
        if self._is_bullish_engulfing():
            return True

        # Bullish Pin Bar (Hammer)
        body_size = abs(self.data.Close[-1] - self.data.Open[-1])
        lower_wick = self.data.Open[-1] - self.data.Low[-1] if self.data.Close[-1] > self.data.Open[-1] else self.data.Close[-1] - self.data.Low[-1]
        upper_wick = self.data.High[-1] - self.data.Close[-1] if self.data.Close[-1] > self.data.Open[-1] else self.data.High[-1] - self.data.Open[-1]
        if body_size > 0 and lower_wick > body_size * 2 and upper_wick < body_size:
            return True

        return False

    def _is_bearish_confirmation(self):
        """Checks for a bearish engulfing or pin bar pattern."""
        if len(self.data) < 2: return False

        # Bearish Engulfing
        if self._is_bearish_engulfing():
            return True

        # Bearish Pin Bar (Shooting Star)
        body_size = abs(self.data.Close[-1] - self.data.Open[-1])
        upper_wick = self.data.High[-1] - self.data.Open[-1] if self.data.Close[-1] < self.data.Open[-1] else self.data.High[-1] - self.data.Close[-1]
        lower_wick = self.data.Close[-1] - self.data.Low[-1] if self.data.Close[-1] < self.data.Open[-1] else self.data.Open[-1] - self.data.Low[-1]
        if body_size > 0 and upper_wick > body_size * 2 and lower_wick < body_size:
            return True

        return False

    def _is_bullish_engulfing(self):
        prev_open, prev_close = self.data.Open[-2], self.data.Close[-2]
        last_open, last_close = self.data.Open[-1], self.data.Close[-1]
        return (prev_close < prev_open and last_close > last_open and
                last_close >= prev_open and last_open <= prev_close)

    def _is_bearish_engulfing(self):
        prev_open, prev_close = self.data.Open[-2], self.data.Close[-2]
        last_open, last_close = self.data.Open[-1], self.data.Close[-1]
        return (prev_close > prev_open and last_close < last_open and
                last_close <= prev_open and last_open >= prev_close)

    def next(self):
        if self.position:
            return

        price = self.data.Close[-1]
        low = self.data.Low[-1]
        high = self.data.High[-1]

        is_uptrend = price > self.ema[-1]
        is_downtrend = price < self.ema[-1]

        if is_uptrend:
            recent_high = self.data.High[-self.lookback_period:].max()
            recent_low = self.data.Low[-self.lookback_period:].min()
            swing_range = recent_high - recent_low
            if swing_range == 0: return

            fib_min = recent_high - swing_range * max(self.fib_levels)
            fib_max = recent_high - swing_range * min(self.fib_levels)

            near_poc = abs(low - self.prev_day_poc[-1]) / self.prev_day_poc[-1] < self.confluence_prox_pct
            near_hvn = abs(low - self.hvn_from_swing_low[-1]) / self.hvn_from_swing_low[-1] < self.confluence_prox_pct
            in_fib = fib_min <= low <= fib_max

            if (near_poc + near_hvn + in_fib) >= 2 and self._is_bullish_confirmation():
                sl = low * (1 - self.sl_buffer_pct)
                tp = self.swing_high_sr[-1]
                if tp > price and (price - sl) > 0 and (tp - price) / (price - sl) >= self.min_rr:
                    self.buy(sl=sl, tp=tp)

        elif is_downtrend:
            recent_high = self.data.High[-self.lookback_period:].max()
            recent_low = self.data.Low[-self.lookback_period:].min()
            swing_range = recent_high - recent_low
            if swing_range == 0: return

            fib_min = recent_low + swing_range * min(self.fib_levels)
            fib_max = recent_low + swing_range * max(self.fib_levels)

            near_poc = abs(high - self.prev_day_poc[-1]) / self.prev_day_poc[-1] < self.confluence_prox_pct
            near_hvn = abs(high - self.hvn_from_swing_high[-1]) / self.hvn_from_swing_high[-1] < self.confluence_prox_pct
            in_fib = fib_min <= high <= fib_max

            if (near_poc + near_hvn + in_fib) >= 2 and self._is_bearish_confirmation():
                sl = high * (1 + self.sl_buffer_pct)
                tp = self.swing_low_sr[-1]
                if tp < price and (sl - price) > 0 and (price - tp) / (sl - price) >= self.min_rr:
                    self.sell(sl=sl, tp=tp)


def get_poc(df_slice, n_bins=50):
    """Calculates the Point of Control for a given DataFrame slice."""
    if df_slice.empty or df_slice['Volume'].sum() == 0:
        return np.nan
    min_price = df_slice['Low'].min()
    max_price = df_slice['High'].max()

    if max_price == min_price:
        return min_price

    bins = np.linspace(min_price, max_price, n_bins)

    price_bins = pd.cut(df_slice['Close'], bins=bins, labels=False, include_lowest=True)
    volume_per_bin = df_slice['Volume'].groupby(price_bins).sum()

    if volume_per_bin.empty:
        return np.nan

    poc_bin_index = volume_per_bin.idxmax()

    poc_price_start = bins[poc_bin_index]
    poc_price_end = bins[poc_bin_index + 1] if poc_bin_index + 1 < len(bins) else max_price

    return (poc_price_start + poc_price_end) / 2


def preprocess_data(data, swing_distance=50, avp_lookback=100, poc_bins=50):
    """
    Pre-processes data to add true daily POC and Anchored Volume Profile HVNs.
    """
    # --- 1. Calculate Previous Day's POC ---
    daily_groups = data.groupby(data.index.date)
    daily_poc = daily_groups.apply(get_poc, n_bins=poc_bins)
    prev_day_poc_map = daily_poc.shift(1).to_dict()
    data['prev_day_poc'] = pd.Series(data.index.date, index=data.index).map(prev_day_poc_map)
    data['prev_day_poc'].ffill(inplace=True)

    # --- 2. Identify Swing Points & S/R Levels ---
    high_peaks, _ = find_peaks(data['High'], distance=swing_distance, prominence=data['Close'].std() * 0.5)
    low_peaks, _ = find_peaks(-data['Low'], distance=swing_distance, prominence=data['Close'].std() * 0.5)

    data['swing_high_sr'] = np.nan
    data.iloc[high_peaks, data.columns.get_loc('swing_high_sr')] = data.iloc[high_peaks]['High']
    data['swing_high_sr'].ffill(inplace=True)

    data['swing_low_sr'] = np.nan
    data.iloc[low_peaks, data.columns.get_loc('swing_low_sr')] = data.iloc[low_peaks]['Low']
    data['swing_low_sr'].ffill(inplace=True)

    # --- 3. Calculate Anchored HVNs (as local POCs) ---
    data['hvn_from_swing_high'] = np.nan
    data['hvn_from_swing_low'] = np.nan

    for peak_idx in high_peaks:
        avp_slice = data.iloc[peak_idx : peak_idx + avp_lookback]
        hvn = get_poc(avp_slice, n_bins=poc_bins)
        if hvn is not None:
             data.iloc[peak_idx:, data.columns.get_loc('hvn_from_swing_high')] = hvn

    for peak_idx in low_peaks:
        avp_slice = data.iloc[peak_idx : peak_idx + avp_lookback]
        hvn = get_poc(avp_slice, n_bins=poc_bins)
        if hvn is not None:
            data.iloc[peak_idx:, data.columns.get_loc('hvn_from_swing_low')] = hvn

    # --- 4. Finalize ---
    return data.dropna()


def sanitize_stats_for_json(stats):
    """
    Sanitizes the backtesting stats object to make it JSON serializable.
    Removes non-serializable types like DataFrames and converts others to compatible formats.
    """
    if stats is None:
        return {}

    # Convert pandas Series to dictionary
    sanitized = stats.to_dict()

    # Remove non-serializable items
    sanitized.pop('_strategy', None)
    sanitized.pop('_equity_curve', None)
    sanitized.pop('_trades', None)

    # Convert specific types
    for key, value in sanitized.items():
        if isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        elif pd.isna(value):
            sanitized[key] = None
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)

    return sanitized


if __name__ == '__main__':
    import json
    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    output_filename = 'results/volume_profile_confluence_sniper.html'
    results_json_path = 'results/temp_result.json'

    # --- Data Loading ---
    try:
        # To match user's OHLCV format, we only need the first 6 columns
        data = pd.read_csv(data_path, usecols=range(6), names=['datetime', 'open', 'high', 'low', 'close', 'volume'], header=0)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        # Ensure correct column names for backtesting.py
        data.columns = [col.capitalize() for col in data.columns]

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        data = pd.DataFrame() # Empty dataframe to avoid crashing

    if not data.empty:
        # --- Preprocessing ---
        data = preprocess_data(data.copy())

        # --- Backtesting ---
        bt = Backtest(data, VolumeProfileConfluenceSniper, cash=100_000, commission=.002)
        stats = bt.run()

        # --- Output ---
        print("--- Backtest Results ---")
        print(stats)

        try:
            bt.plot(filename=output_filename, open_browser=False)
            print(f"\nPlot saved to {output_filename}")
        except Exception as e:
            print(f"\nError plotting results: {e}. Continuing without plot.")

        # --- Save results ---
        try:
            sanitized_results = sanitize_stats_for_json(stats)
            with open(results_json_path, 'w') as f:
                json.dump(sanitized_results, f, indent=4)
            print(f"Results saved to {results_json_path}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")
    else:
        print("Could not run backtest because data is empty.")
