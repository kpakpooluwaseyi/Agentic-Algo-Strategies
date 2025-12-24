
import numpy as np
import pandas as pd
from backtesting import Strategy
from scipy.signal import find_peaks

def find_swings(array, distance):
    """
    Custom indicator function to find swing highs and lows using scipy.signal.find_peaks.
    Returns a signal array: 1 for a peak (swing high), -1 for a trough (swing low), 0 otherwise.
    """
    # Find peaks (local maxima) for swing highs
    peaks, _ = find_peaks(array, distance=distance)
    # Find troughs (local minima) by inverting the array
    troughs, _ = find_peaks(-array, distance=distance)

    # Initialize a signal array with zeros
    signals = np.zeros(len(array))
    # Mark peaks with 1
    signals[peaks] = 1
    # Mark troughs with -1
    signals[troughs] = -1
    return signals

class FibonacciRetracementStrategy(Strategy):
    """
    A strategy that identifies swing points, waits for a retracement to a specified
    Fibonacci level, and enters a trade in the direction of the original trend.
    """
    # --- Optimizable Parameters ---
    # `peak_distance`: The minimum number of bars between consecutive swing points.
    # This helps filter out minor fluctuations and identify more significant swings.
    peak_distance = 20

    # `fib_level`: The Fibonacci retracement level to trigger a trade entry.
    # 0.618 (61.8%) is a key Fibonacci level.
    fib_level = 0.618

    # `sl_buffer_pct`: A percentage buffer added to the stop-loss to place it slightly
    # beyond the swing point, reducing the chance of being stopped out by noise.
    sl_buffer_pct = 0.01 # 1%

    def init(self):
        """
        Initialize the strategy. This method is called once before the backtest starts.
        """
        # Use the custom `find_swings` function as an indicator.
        # self.I() registers the function, and the result is accessible in `self.next()`.
        self.swing_points = self.I(find_swings, self.data.Close, self.peak_distance)

        # --- State Machine Variables ---
        # These variables track the current state of the pattern detection.
        self.swing_high = None  # Stores the price of the last confirmed swing high
        self.swing_low = None   # Stores the price of the last confirmed swing low
        self.setup_direction = 0 # 1 for long setup, -1 for short setup

    def next(self):
        """
        The main strategy logic, called for each bar of the data.
        """
        # If a position is already open, we don't need to check for new entries.
        if self.position:
            return

        # --- Step 1: Detect a new setup if we don't have one ---
        if self.setup_direction == 0:
            swings = np.where(self.swing_points != 0)[0]
            if len(swings) < 2:
                return

            last_swing_idx, prev_swing_idx = swings[-1], swings[-2]

            # A. Detect a short setup: A swing high followed by a swing low
            if self.swing_points[prev_swing_idx] == 1 and self.swing_points[last_swing_idx] == -1:
                self.swing_high = self.data.High[prev_swing_idx]
                self.swing_low = self.data.Low[last_swing_idx]
                self.setup_direction = -1
            # B. Detect a long setup: A swing low followed by a swing high
            elif self.swing_points[prev_swing_idx] == -1 and self.swing_points[last_swing_idx] == 1:
                self.swing_low = self.data.Low[prev_swing_idx]
                self.swing_high = self.data.High[last_swing_idx]
                self.setup_direction = 1

        # --- Step 2: Process the active setup ---
        if self.setup_direction == 1:  # Active long setup
            # Invalidation: If price makes a new high, the setup is void.
            if self.data.High[-1] > self.swing_high:
                self.setup_direction = 0
                return

            fib_entry_level = self.swing_high - (self.swing_high - self.swing_low) * self.fib_level

            # Entry: If price pulls back to the fib level
            if self.data.Low[-1] <= fib_entry_level:
                stop_loss = self.swing_low * (1 - self.sl_buffer_pct)
                take_profit = self.swing_high

                # Final validation before placing order to prevent ValueError
                if self.data.Close[-1] > stop_loss and self.data.Close[-1] < take_profit:
                    self.buy(sl=stop_loss, tp=take_profit)

                # Setup is now consumed or invalid, so reset.
                self.setup_direction = 0

        elif self.setup_direction == -1:  # Active short setup
            # Invalidation: If price makes a new low, the setup is void.
            if self.data.Low[-1] < self.swing_low:
                self.setup_direction = 0
                return

            fib_entry_level = self.swing_low + (self.swing_high - self.swing_low) * self.fib_level

            # Entry: If price pulls back to the fib level
            if self.data.High[-1] >= fib_entry_level:
                stop_loss = self.swing_high * (1 + self.sl_buffer_pct)
                take_profit = self.swing_low

                # Final validation before placing order to prevent ValueError
                if self.data.Close[-1] < stop_loss and self.data.Close[-1] > take_profit:
                    self.sell(sl=stop_loss, tp=take_profit)

                # Setup is now consumed or invalid, so reset.
                self.setup_direction = 0


if __name__ == '__main__':
    import os
    import json
    from backtesting import Backtest

    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # --- Data Cleaning ---
        # Ensure column names are capitalized as required by backtesting.py
        data.columns = [c.strip().title() for c in data.columns]
        # Drop 'Unnamed' columns if they exist
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        # Ensure index is a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        print("Running backtest with default parameters...")
        bt = Backtest(data, FibonacciRetracementStrategy, cash=100_000, commission=.002)

        stats = bt.run()
        print(stats)

        # --- Save Results ---
        # Ensure the results directory exists
        os.makedirs('results', exist_ok=True)

        # Sanitize the stats object for JSON serialization
        def sanitize_stats(stats_obj):
            sanitized = {}
            for key, value in stats_obj.items():
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
            # Exclude non-serializable objects like the strategy instance
            sanitized.pop('_strategy', None)
            sanitized.pop('_equity_curve', None)
            sanitized.pop('_trades', None)
            return sanitized

        results_dict = sanitize_stats(stats)

        # Save to JSON
        json_path = 'results/temp_result.json'
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f"Backtest statistics saved to {json_path}")

        # --- Generate Plot ---
        plot_path = 'results/strategy_06f7f6730c08.html'
        try:
            bt.plot(filename=plot_path, open_browser=False)
            print(f"Backtest plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
