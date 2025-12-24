
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply
import numpy as np
import json
import os
from scipy.signal import find_peaks

def sanitize_stats(stats):
    """
    Sanitizes the stats object from backtesting.py to be JSON serializable,
    handling specific numpy and pandas types.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value)
        elif isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        # Add other type conversions as needed
    # Don't include the strategy object or trades/equity curve data in JSON
    if '_strategy' in sanitized:
        del sanitized['_strategy']
    if '_trades' in sanitized:
        del sanitized['_trades']
    if '_equity_curve' in sanitized:
        del sanitized['_equity_curve']
    return sanitized


def EMA(series, period):
    """Custom EMA function using pandas_ta."""
    return talib.EMA(series, timeperiod=period)

class ReversalOnEma200(Strategy):
    # Optimizable parameters
    pattern_lookback = 50
    ema_proximity_pct = 0.005 # 0.5%
    sl_buffer_pct = 0.01 # 1%
    risk_reward_ratio = 2.0

    def init(self):
        # Initialize the 1-hour 200 EMA
        self.ema200_1h = resample_apply('1H', EMA, self.data.Close, 200)

    def next(self):
        # Do not proceed if there are not enough bars
        if len(self.data.Close) < self.pattern_lookback:
            return

        # In this strategy, exit logic is handled by SL/TP orders, so we don't
        # need active position management. We only check for entries if there's no open position.
        if not self.position:
            self._detect_m_pattern_and_enter()
            self._detect_w_pattern_and_enter()

    def _detect_m_pattern_and_enter(self):
        """Detects an M-pattern and places a short order."""
        highs = self.data.High[-self.pattern_lookback:]
        peaks, _ = find_peaks(highs, distance=5)

        if len(peaks) < 2:
            return

        peak2_idx, peak1_idx = peaks[-1], peaks[-2]
        peak2_price, peak1_price = highs[peak2_idx], highs[peak1_idx]

        trough_slice = self.data.Low[peak1_idx:peak2_idx]
        if len(trough_slice) == 0:
            return
        center_trough_price = np.min(trough_slice)

        # Check for M-pattern confirmation (break below center trough)
        if self.data.Close[-1] < center_trough_price:
            # Check proximity of the highest peak to the EMA
            highest_peak = max(peak1_price, peak2_price)
            ema_value_at_peak = self.ema200_1h[-(self.pattern_lookback - (peak1_idx if peak1_price > peak2_price else peak2_idx))]

            is_near_ema = abs(highest_peak - ema_value_at_peak) / ema_value_at_peak <= self.ema_proximity_pct

            if is_near_ema:
                entry_price = self.data.Close[-1]
                stop_loss = highest_peak * (1 + self.sl_buffer_pct)

                # Ensure SL is above entry for a short
                if stop_loss > entry_price:
                    take_profit = entry_price - (stop_loss - entry_price) * self.risk_reward_ratio
                    if take_profit > 0: # Ensure TP is valid
                        self.sell(sl=stop_loss, tp=take_profit)

    def _detect_w_pattern_and_enter(self):
        """Detects a W-pattern and places a long order."""
        lows = self.data.Low[-self.pattern_lookback:]
        troughs, _ = find_peaks(-lows, distance=5)

        if len(troughs) < 2:
            return

        trough2_idx, trough1_idx = troughs[-1], troughs[-2]
        trough2_price, trough1_price = lows[trough2_idx], lows[trough1_idx]

        peak_slice = self.data.High[trough1_idx:trough2_idx]
        if len(peak_slice) == 0:
            return
        center_peak_price = np.max(peak_slice)

        # Check for W-pattern confirmation (break above center peak)
        if self.data.Close[-1] > center_peak_price:
            # Check proximity of the lowest trough to the EMA
            lowest_trough = min(trough1_price, trough2_price)
            ema_value_at_trough = self.ema200_1h[-(self.pattern_lookback - (trough1_idx if trough1_price < trough2_price else trough2_idx))]

            is_near_ema = abs(lowest_trough - ema_value_at_trough) / ema_value_at_trough <= self.ema_proximity_pct

            if is_near_ema:
                entry_price = self.data.Close[-1]
                stop_loss = lowest_trough * (1 - self.sl_buffer_pct)

                # Ensure SL is below entry for a long
                if stop_loss < entry_price:
                    take_profit = entry_price + (entry_price - stop_loss) * self.risk_reward_ratio
                    self.buy(sl=stop_loss, tp=take_profit)

def load_data(filepath, start_date=None, end_date=None):
    """Loads and preprocesses CSV data."""
    data = pd.read_csv(
        filepath,
        index_col=0,
        parse_dates=True,
        skipinitialspace=True
    )
    # Clean column names (strip whitespace, title case, remove trailing commas)
    data.columns = [c.strip().title().replace(',', '') for c in data.columns]
    data = data.rename(columns={'Datetime': 'datetime'})

    # Drop unnamed columns that might be created by trailing commas
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Ensure required columns are present
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"CSV file must contain the following columns: {required_cols}")

    # Ensure index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    data = data.sort_index()

    if start_date:
        data = data[data.index >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data.index <= pd.to_datetime(end_date)]

    return data

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        print("Data file not found. Generating synthetic data for demonstration.")
        from backtesting.test import GOOG
        data = GOOG.copy().iloc[-1000:] # Smaller dataset for synthetic
    else:
        # Load a smaller, more recent slice of data to speed up optimization
        data = load_data(data_path, start_date='2023-01-01', end_date='2023-06-01')

    bt = Backtest(data, ReversalOnEma200, cash=100_000, commission=.002, finalize_trades=True)

    # --- Optimization ---
    print("\n--- Running optimization ---")
    stats_optimized = bt.optimize(
        pattern_lookback=range(40, 70, 10),            # Narrowed range
        ema_proximity_pct=[0.005, 0.007, 0.01],         # Fewer values
        sl_buffer_pct=[0.01, 0.015],                    # Fewer values
        risk_reward_ratio=[1.5, 2.0, 2.5],              # Fewer values
        maximize='Sharpe Ratio',
        constraint=lambda p: p.pattern_lookback > 20,
        max_tries=50 # Limit the number of combinations
    )
    print("\n--- Best optimization results ---")
    print(stats_optimized)

    # --- Final Run with Best Parameters and Saving Results ---
    # The stats object from optimize() is the result of the best run
    final_stats = stats_optimized

    # Sanitize and save results to temp_result.json
    os.makedirs('results', exist_ok=True)
    sanitized_stats = sanitize_stats(final_stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)
    print("\nSaved optimized stats to results/temp_result.json")

    # Generate and save plot for the best run
    plot_filename = 'results/reversal_on_ema_200__aka__hold_the_mayo__.html'
    print(f"Saving plot of best run to {plot_filename}...")
    try:
        # bt.plot() will automatically use the results from the last run (the optimization)
        bt.plot(filename=plot_filename, open_browser=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")
