
import json
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply
from scipy.signal import find_peaks

# --- Indicator and Utility Functions ---

def EMA(series, n):
    """Exponential Moving Average"""
    return pd.Series(series).ewm(span=n, adjust=False).mean().values

# --- Data Pre-processing ---

def preprocess_data(df: pd.DataFrame, prominence_15m: int = 10, ema_period=200):
    """
    Performs multi-timeframe analysis by identifying signals on a 15M timeframe
    and merging them into the original 1M DataFrame.
    """
    # -- Higher Timeframe (HTF) Analysis (15M) --
    df_15m = df['Close'].resample('15min').ohlc()

    # Find peaks (swing highs) and troughs (swing lows) on the 15M chart
    peak_indices, _ = find_peaks(df_15m['high'], prominence=prominence_15m)
    trough_indices, _ = find_peaks(-df_15m['low'], prominence=prominence_15m)

    df_15m['peak'] = False
    df_15m.iloc[peak_indices, df_15m.columns.get_loc('peak')] = True
    df_15m['trough'] = False
    df_15m.iloc[trough_indices, df_15m.columns.get_loc('trough')] = True

    # Identify valid setups: a peak followed by a trough
    df_15m['setup_high'] = np.nan
    df_15m['setup_low'] = np.nan

    last_peak_high = np.nan
    for i in range(len(df_15m)):
        if df_15m.iloc[i]['peak']:
            last_peak_high = df_15m.iloc[i]['high']
        if df_15m.iloc[i]['trough'] and not pd.isna(last_peak_high):
            # This trough is the low after the last peak
            df_15m.iloc[i, df_15m.columns.get_loc('setup_high')] = last_peak_high
            df_15m.iloc[i, df_15m.columns.get_loc('setup_low')] = df_15m.iloc[i]['low']
            last_peak_high = np.nan # Reset after finding a pair

    # Calculate the 50% Fibonacci level for each valid setup
    df_15m['fib_50'] = df_15m['setup_high'] - 0.5 * (df_15m['setup_high'] - df_15m['setup_low'])

    # -- Merge HTF signals back into the 1M DataFrame --

    # Select only the necessary columns to merge
    htf_signals = df_15m[['fib_50', 'setup_low']].copy()
    htf_signals.dropna(inplace=True)

    # Use merge_asof to map the 15M signals to the 1M bars
    df = pd.merge_asof(
        df, htf_signals,
        left_index=True,
        right_index=True,
        direction='backward'
    )
    df[['fib_50', 'setup_low']] = df[['fib_50', 'setup_low']].ffill()

    # -- Session and Other Indicator Calculations --

    # Asia Session (e.g., 00:00 - 08:00 UTC)
    df['is_asia'] = (df.index.hour >= 0) & (df.index.hour < 8)
    asia_high = df[df['is_asia']].groupby(df[df['is_asia']].index.date)['High'].transform('max')
    asia_low = df[df['is_asia']].groupby(df[df['is_asia']].index.date)['Low'].transform('min')
    df['asia_mid'] = (asia_high + asia_low) / 2

    # Daily High/Low
    df['lod'] = df.groupby(df.index.date)['Low'].transform('min')

    # EMA
    df['ema'] = EMA(df['Close'], n=ema_period)

    df.dropna(inplace=True)

    return df

# --- Synthetic Data Generation ---

def _inject_pattern(df, start_idx):
    """Helper to inject a single Drop-Retrace-Reverse pattern."""
    # 1. The Drop
    drop_duration = np.random.randint(180, 360) # 3-6 hours
    drop_end_idx = start_idx + drop_duration
    if drop_end_idx >= len(df): return

    start_price = df.iloc[start_idx]['Close']
    drop_amount = start_price * np.random.uniform(0.01, 0.03) # 1-3% drop
    end_price = start_price - drop_amount

    drop_prices = np.linspace(start_price, end_price, drop_duration)
    noise = np.random.normal(0, 0.0001, drop_duration).cumsum()
    close_col_idx = df.columns.get_loc('Close')
    df.iloc[start_idx:drop_end_idx, close_col_idx] = drop_prices + noise

    # 2. The Retracement
    retrace_duration = np.random.randint(120, 240) # 2-4 hours
    retrace_end_idx = drop_end_idx + retrace_duration
    if retrace_end_idx >= len(df): return

    high_of_drop = df.iloc[start_idx:drop_end_idx]['Close'].max()
    low_of_drop = df.iloc[start_idx:drop_end_idx]['Close'].min()
    fib_50_level = high_of_drop - 0.5 * (high_of_drop - low_of_drop)

    retrace_prices = np.linspace(low_of_drop, fib_50_level, retrace_duration)
    noise = np.random.normal(0, 0.00005, retrace_duration).cumsum()
    df.iloc[drop_end_idx:retrace_end_idx, close_col_idx] = retrace_prices + noise

    # Create a bearish engulfing/pin bar at the peak
    peak_idx = retrace_end_idx - 1
    df.loc[df.index[peak_idx], 'High'] = fib_50_level * 1.0005
    df.loc[df.index[peak_idx], 'Open'] = fib_50_level * 0.9998
    df.loc[df.index[peak_idx], 'Close'] = fib_50_level * 0.9990
    df.loc[df.index[peak_idx], 'Low'] = fib_50_level * 0.9989

    # 3. The Reversal (subsequent drop)
    reversal_duration = np.random.randint(180, 360)
    reversal_end_idx = retrace_end_idx + reversal_duration
    if reversal_end_idx >= len(df): return

    reversal_drop_amount = drop_amount * np.random.uniform(0.6, 1.2)
    reversal_end_price = fib_50_level - reversal_drop_amount

    reversal_prices = np.linspace(fib_50_level, reversal_end_price, reversal_duration)
    noise = np.random.normal(0, 0.0001, reversal_duration).cumsum()
    df.iloc[retrace_end_idx:reversal_end_idx, close_col_idx] = reversal_prices + noise

def generate_synthetic_data(days=90, num_patterns=15):
    """
    Generates synthetic 1M OHLCV data that includes the specific patterns
    the Fibonacci Center Peak Scalp strategy looks for.
    """
    rng = np.random.default_rng(42)
    n_minutes = days * 24 * 60
    index = pd.date_range(start='2023-01-01', periods=n_minutes, freq='min')

    # Base random walk
    base_price = 1000
    returns = rng.normal(loc=0, scale=0.00005, size=n_minutes)
    price = base_price * (1 + returns).cumprod()

    df = pd.DataFrame(index=index)
    df['Close'] = price

    # Inject patterns at random locations
    injection_points = rng.choice(len(df) - 720, size=num_patterns, replace=False)
    for idx in sorted(injection_points):
        _inject_pattern(df, idx)

    # Generate OHLC from Close
    df['Open'] = df['Close'].shift(1)
    df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0, 0.0005, size=len(df)) * df['Close']
    df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0, 0.0005, size=len(df)) * df['Close']
    df['Volume'] = rng.integers(100, 10000, size=len(df))
    df.iloc[0, df.columns.get_loc('Open')] = df.iloc[0]['Close'] # Set first Open

    df.dropna(inplace=True)
    return df

# --- Strategy Class ---

class FibonacciCenterPeakScalpStrategy(Strategy):
    """
    A multi-timeframe strategy that identifies significant price drops on a 15M chart,
    waits for a 50% Fibonacci retracement, and enters a short position on a 1M
    bearish confirmation.
    """
    # Optimization parameters (will be set by the optimizer)
    min_rr = 5
    confirmation_window = 3

    # --- State tracking variables ---
    # These variables are crucial for the state machine logic in `next()`

    # `setup_fib_level` stores the active 50% Fib level we are monitoring.
    # It's NaN if no valid setup is currently being tracked.
    setup_fib_level = np.nan

    # `setup_low` stores the low of the 15M drop, needed for TP calculation.
    setup_low = np.nan

    # `monitoring_for_confirmation` becomes true when the price touches the Fib level.
    monitoring_for_confirmation = False

    # `confirmation_candle_index` stores the bar number when monitoring starts.
    confirmation_candle_index = 0

    # `last_trade_was_short_tp` is for the "Bag Flip" logic.
    last_trade_was_short_tp = False

    # --- Active Trade Management State ---
    # These are set when a trade is opened and cleared on close.
    initial_tp = np.nan
    extended_target = np.nan
    is_extended_trade = False

    def init(self):
        identity_func = lambda x: x
        self.fib_50 = self.I(identity_func, self.data.df['fib_50'].values)
        self.setup_low_indicator = self.I(identity_func, self.data.df['setup_low'].values)
        self.ema = self.I(identity_func, self.data.df['ema'].values)
        self.lod = self.I(identity_func, self.data.df['lod'].values)

    def next(self):
        price = self.data.Close[-1]

        # --- Refactored Exit Logic (Manual Management) ---
        if self.position.is_short:
            # Check for strong bullish reversal at the initial TP level
            if not self.is_extended_trade and price <= self.initial_tp:
                is_green_candle = self.data.Close[-1] > self.data.Open[-1]
                candle_range = self.data.High[-1] - self.data.Low[-1]
                body_size = self.data.Close[-1] - self.data.Open[-1]
                is_strong_bullish = is_green_candle and candle_range > 0 and (body_size / candle_range) > 0.3

                if is_strong_bullish:
                    # Close the trade and trigger the bag flip
                    self.position.close()
                    self.last_trade_was_short_tp = True
                    return
                else:
                    # No buying pressure, hold for the extended target
                    self.is_extended_trade = True
                    self.extended_target = self.lod[-1] # Target Low of Day
                    # print(f"DEBUG [{self.data.index[-1]}]: Holding short for extended target {self.extended_target:.2f}")

            # Check if the extended target has been hit
            if self.is_extended_trade and price <= self.extended_target:
                self.position.close()
                return

        # --- Bag Flip Logic ---
        if self.last_trade_was_short_tp:
            self.last_trade_was_short_tp = False
            self.buy(size=1)
            return

        # --- Entry Logic ---
        if not self.position and not self.monitoring_for_confirmation:
            current_fib = self.fib_50[-1]
            if not np.isnan(current_fib):
                self.setup_fib_level = current_fib
                self.setup_low = self.setup_low_indicator[-1]
                if abs(price - self.setup_fib_level) / price < 0.005:
                    self.monitoring_for_confirmation = True
                    self.confirmation_candle_index = len(self.data.Close)

        if self.monitoring_for_confirmation:
            if abs(price - self.setup_fib_level) / price > 0.01 or \
               len(self.data.Close) > self.confirmation_candle_index + self.confirmation_window:
                self.monitoring_for_confirmation = False
                return

            is_red_candle = self.data.Close[-1] < self.data.Open[-1]
            candle_range = self.data.High[-1] - self.data.Low[-1]
            body_size = self.data.Open[-1] - self.data.Close[-1]
            is_strong_bearish = is_red_candle and candle_range > 0 and (body_size / candle_range) > 0.3

            if is_strong_bearish:
                entry_price = self.data.Close[-1]
                stop_loss = self.data.High[-1] * 1.0005
                self.initial_tp = entry_price - 0.5 * (entry_price - self.setup_low)

                if stop_loss <= entry_price or self.initial_tp >= entry_price:
                    return

                risk = abs(entry_price - stop_loss)
                reward = abs(entry_price - self.initial_tp)

                if risk == 0: return
                rr_ratio = reward / risk
                ema_confluence = abs(self.setup_fib_level - self.ema[-1]) / self.setup_fib_level < 0.005

                if rr_ratio >= self.min_rr and ema_confluence:
                    self.sell(size=1, sl=stop_loss) # Remove hard TP
                    self.monitoring_for_confirmation = False
                    self.is_extended_trade = False # Reset trade state

# --- Main Execution Block ---

if __name__ == '__main__':
    # 1. Generate or Load Data
    raw_data = generate_synthetic_data(days=90) # Reduced days for faster optimization

    # Ensure results directory exists
    import os
    os.makedirs('results', exist_ok=True)

    # --- Custom Optimization Loop ---
    # backtesting.py doesn't support optimizing parameters used in pre-processing.
    # We must manually loop through those and run `bt.optimize` on the rest.

    best_stats = None
    best_ema_period = 0

    # Define parameter ranges
    ema_periods_to_test = range(100, 301, 100)
    prominence_levels_to_test = [10, 20] # Example of another pre-proc param

    print(f"Starting custom optimization loop...")
    for prominence in prominence_levels_to_test:
        for ema_period in ema_periods_to_test:
            print(f"\n--- Testing Pre-processing Params: Prominence={prominence}, EMA={ema_period} ---")

            # 2. Pre-process Data with the current loop's parameters
            processed_data = preprocess_data(
                raw_data.copy(),
                prominence_15m=prominence,
                ema_period=ema_period
            )

            if processed_data.empty:
                print("Warning: Pre-processing resulted in an empty DataFrame. Skipping this run.")
                continue

            # 3. Run Backtest and Optimize strategy-level parameters
            bt = Backtest(processed_data, FibonacciCenterPeakScalpStrategy, cash=100_000, commission=.001)

            stats = bt.optimize(
                min_rr=[3, 5],
                confirmation_window=range(3, 6),
                maximize='Sharpe Ratio',
                constraint=lambda p: p.min_rr > 0
            )

            # Manually add the pre-processing param to the results for tracking
            stats['_ema_period'] = ema_period
            stats['_prominence'] = prominence

            print(f"Stats for this run:\n{stats[['Sharpe Ratio', '# Trades']]}")

            # 4. Track the Best Overall Result
            if best_stats is None or stats['Sharpe Ratio'] > best_stats['Sharpe Ratio']:
                print(f"*** New best result found! Sharpe: {stats['Sharpe Ratio']:.2f} ***")
                best_stats = stats

    print("\n--- Optimization Complete ---")
    print("Best overall stats:\n", best_stats)

    # 5. Save the best results
    stats = best_stats
    results = {
        'strategy_name': 'fibonacci_center_peak_scalp',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0),
        'best_params': {
             # Extract best strategy-level params from the winning strategy instance
            'min_rr': stats._strategy.min_rr,
            'confirmation_window': stats._strategy.confirmation_window,
             # Add the best pre-processing params
            'ema_period': stats._ema_period,
            'prominence': stats._prominence
        }
    }

    def sanitize_for_json(obj):
        """Recursively convert numpy types to native Python types for JSON."""
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(i) for i in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

    sanitized_results = sanitize_for_json(results)

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_results, f, indent=2)

    print("\nResults saved to results/temp_result.json")

    # 6. Generate Plot
    try:
        bt.plot(filename='results/fibonacci_center_peak_scalp_plot.html', open_browser=False)
        print("Plot saved to results/fibonacci_center_peak_scalp_plot.html")
    except TypeError as e:
        print(f"\nCould not generate plot due to a known issue with the library: {e}")
        print("This does not affect the JSON results.")
