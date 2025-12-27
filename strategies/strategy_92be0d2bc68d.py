from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import json

def preprocess_data_with_swing_points(df, distance=50, prominence_pct=0.02):
    """
    Calculates swing points and adds them as columns to the DataFrame.
    A swing point is marked with its price, otherwise NaN.
    """
    df_copy = df.copy()
    close_series = df_copy['Close']
    price_range = close_series.max() - close_series.min()
    prominence = price_range * prominence_pct

    high_peaks_indices, _ = find_peaks(close_series, distance=distance, prominence=prominence)
    low_peaks_indices, _ = find_peaks(-close_series, distance=distance, prominence=prominence)

    df_copy['swing_high'] = np.nan
    df_copy.iloc[high_peaks_indices, df_copy.columns.get_loc('swing_high')] = close_series.iloc[high_peaks_indices]

    df_copy['swing_low'] = np.nan
    df_copy.iloc[low_peaks_indices, df_copy.columns.get_loc('swing_low')] = close_series.iloc[low_peaks_indices]

    return df_copy

def passthrough_indicator(series):
    """A simple function to pass a pre-calculated series to the strategy."""
    return series

class FibonacciReversalExtension(Strategy):
    # --- Optimizable Parameters ---
    peak_distance = 20 # More sensitive swing detection
    peak_prominence_pct = 0.01 # More sensitive swing detection
    entry_retracement_levels = (0.382, 0.5, 0.618)
    take_profit_extension = 1.618
    stop_loss_retracement = 1.0 # Stop at 100% retracement of the impulse wave

    def init(self):
        # Indicators for swing points from pre-processed data
        self.swing_highs = self.I(passthrough_indicator, self.data.df['swing_high'])
        self.swing_lows = self.I(passthrough_indicator, self.data.df['swing_low'])

        # State machine variables
        self.recent_swings = []
        self.impulse_wave = None
        self.entry_level_hit = None

    def next(self):
        current_price = self.data.Close[-1]

        # --- Trade Management ---
        if self.position:
            return

        # --- State Reset on Trade Completion/Miss ---
        if self.entry_level_hit:
            self.impulse_wave = None
            self.entry_level_hit = None

        # --- Detect and Store Alternating Swing Points ---
        if not np.isnan(self.swing_highs[-1]):
            new_swing = ('high', self.swing_highs[-1])
            if not self.recent_swings or self.recent_swings[-1][0] != 'high':
                self.recent_swings.append(new_swing)
        elif not np.isnan(self.swing_lows[-1]):
            new_swing = ('low', self.swing_lows[-1])
            if not self.recent_swings or self.recent_swings[-1][0] != 'low':
                self.recent_swings.append(new_swing)

        if len(self.recent_swings) > 5:
            self.recent_swings.pop(0)

        # --- Invalidation Logic ---
        if self.impulse_wave:
            start_price = self.impulse_wave['start']
            if self.impulse_wave['dir'] == 'long' and current_price < start_price:
                self.impulse_wave = None
            elif self.impulse_wave['dir'] == 'short' and current_price > start_price:
                self.impulse_wave = None

        # --- Identify New Impulse Wave ---
        if self.impulse_wave is None and len(self.recent_swings) >= 2:
            last_swing_type, last_swing_price = self.recent_swings[-1]
            prev_swing_type, prev_swing_price = self.recent_swings[-2]

            if last_swing_type == 'high' and prev_swing_type == 'low':
                self.impulse_wave = {'start': prev_swing_price, 'end': last_swing_price, 'dir': 'long'}
            elif last_swing_type == 'low' and prev_swing_type == 'high':
                self.impulse_wave = {'start': prev_swing_price, 'end': last_swing_price, 'dir': 'short'}

        # --- Check for Retracement ---
        if self.impulse_wave and self.entry_level_hit is None:
            start, end = self.impulse_wave['start'], self.impulse_wave['end']
            price_range = abs(end - start)

            for level in self.entry_retracement_levels:
                retracement_price = end - price_range * level if self.impulse_wave['dir'] == 'long' else end + price_range * level

                if (self.impulse_wave['dir'] == 'long' and current_price <= retracement_price) or \
                   (self.impulse_wave['dir'] == 'short' and current_price >= retracement_price):
                    self.entry_level_hit = retracement_price
                    break

        # --- Entry Confirmation and Execution ---
        if self.entry_level_hit and not self.position:
            start, end = self.impulse_wave['start'], self.impulse_wave['end']
            price_range = abs(end - start)

            if self.impulse_wave['dir'] == 'long' and self.data.Close[-1] > self.data.Open[-1]:
                sl = start
                tp = end + price_range * (self.take_profit_extension - 1)
                if tp > current_price and current_price > sl:
                    self.buy(sl=sl, tp=tp)

            elif self.impulse_wave['dir'] == 'short' and self.data.Close[-1] < self.data.Open[-1]:
                sl = start
                tp = end - price_range * (self.take_profit_extension - 1)
                if tp < current_price and current_price < sl:
                    self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Using synthetic data.")
        from backtesting.test import GOOG
        data = GOOG.copy().iloc[-3000:]

    data.columns = [col.strip().title() for col in data.columns]

    # Pre-process data to add swing points
    # Use the strategy's default parameters for consistency
    data = preprocess_data_with_swing_points(data,
                                             distance=FibonacciReversalExtension.peak_distance,
                                             prominence_pct=FibonacciReversalExtension.peak_prominence_pct)

    bt = Backtest(data, FibonacciReversalExtension, cash=100_000, commission=.002)
    stats = bt.run()

    print(stats)

    import os
    os.makedirs('results', exist_ok=True)

    def sanitize_for_json(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if isinstance(obj, pd.Timedelta): return str(obj)
        if isinstance(obj, pd.Series): return sanitize_for_json(obj.to_dict())
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items() if k not in ['_strategy', '_equity_curve', '_trades']}
        return obj

    clean_stats = sanitize_for_json(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(clean_stats, f, indent=4)
    print("Backtest results saved to results/temp_result.json")

    try:
        bt.plot(filename='results/strategy_92be0d2bc68d.html', open_browser=False)
        print("Backtest plot saved to results/strategy_92be0d2bc68d.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
