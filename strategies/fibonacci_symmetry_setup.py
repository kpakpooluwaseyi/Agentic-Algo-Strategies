from backtesting import Backtest, Strategy
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import json

# Custom indicator function to find peaks (swing highs/lows)
def find_swings(array, distance):
    # scipy.signal.find_peaks finds local maxima.
    # To find swing lows, we invert the series.
    peaks, _ = find_peaks(array, distance=distance)
    troughs, _ = find_peaks(-array, distance=distance)

    # Create a signal array: 1 for peak, -1 for trough, 0 otherwise
    signals = np.zeros(len(array))
    signals[peaks] = 1
    signals[troughs] = -1
    return signals

class FibonacciSymmetrySetupStrategy(Strategy):
    # Optimizable parameters
    peak_distance = 10          # Distance for detecting swing points
    symmetry_tolerance = 0.05   # 5% tolerance for the symmetry projection
    sl_buffer = 0.002           # Buffer for stop-loss in percentage (0.2%)
    tp_extension_level = 1.272  # Fibonacci extension level for take-profit
    min_rr = 2.0                # Minimum Risk-to-Reward ratio

    def init(self):
        self.swing_points = self.I(find_swings, self.data.Close, self.peak_distance)
        self.reset_state()

    def reset_state(self):
        """Resets the state of the state machine."""
        self.point_A = None
        self.point_B = None
        self.point_C = None
        self.projection_D = None
        self.setup_direction = 0  # 1 for long, -1 for short

    def next(self):
        # If a position is open, do nothing until it's closed.
        if self.position:
            return

        # === STATE 1: Find a valid A-B-C pattern ===
        if self.setup_direction == 0:
            swings = np.where(self.swing_points != 0)[0]
            if len(swings) < 3:
                return

            c_idx, b_idx, a_idx = swings[-1], swings[-2], swings[-3]

            # Try to find a SHORT setup (downtrend)
            if self.swing_points[a_idx] == -1 and self.swing_points[b_idx] == 1 and self.swing_points[c_idx] == -1:
                a_price, b_price, c_price = self.data.Low[a_idx], self.data.High[b_idx], self.data.Low[c_idx]
                if c_price < a_price: # Confirming downtrend
                    self.point_A, self.point_B, self.point_C = a_price, b_price, c_price
                    self.projection_D = self.point_C + (self.point_B - self.point_A)
                    self.setup_direction = -1

            # Try to find a LONG setup (uptrend)
            elif self.swing_points[a_idx] == 1 and self.swing_points[b_idx] == -1 and self.swing_points[c_idx] == 1:
                a_price, b_price, c_price = self.data.High[a_idx], self.data.Low[b_idx], self.data.High[c_idx]
                if c_price > a_price: # Confirming uptrend
                    self.point_A, self.point_B, self.point_C = a_price, b_price, c_price
                    self.projection_D = self.point_C - (self.point_A - self.point_B)
                    self.setup_direction = 1

        # === STATE 2: Wait for price to test the projection D and show reversal ===
        if self.setup_direction != 0:
            upper_bound = self.projection_D * (1 + self.symmetry_tolerance)
            lower_bound = self.projection_D * (1 - self.symmetry_tolerance)

            # --- Invalidation logic ---
            # If price moves too far from C before hitting D, invalidate
            if self.setup_direction == -1 and self.data.Low[-1] < self.point_C:
                 self.reset_state()
                 return
            if self.setup_direction == 1 and self.data.High[-1] > self.point_C:
                 self.reset_state()
                 return

            # --- Entry logic ---
            if self.setup_direction == -1: # Short setup active
                if lower_bound <= self.data.High[-1] <= upper_bound:
                    is_bullish_prev = self.data.Close[-2] > self.data.Open[-2]
                    is_bearish_curr = self.data.Close[-1] < self.data.Open[-1]
                    is_engulfing = self.data.Open[-1] >= self.data.Close[-2] and self.data.Close[-1] <= self.data.Open[-2]

                    if is_bullish_prev and is_bearish_curr and is_engulfing:
                        entry_price = self.data.Close[-1]
                        stop_loss = self.data.High[-1] * (1 + self.sl_buffer)
                        take_profit = self.data.High[-1] - (self.data.High[-1] - self.point_C) * self.tp_extension_level

                        if stop_loss > entry_price and take_profit < entry_price:
                            rr = (entry_price - take_profit) / (stop_loss - entry_price)
                            if rr >= self.min_rr:
                                self.sell(sl=stop_loss, tp=take_profit)
                        self.reset_state() # Reset after attempting a trade

            elif self.setup_direction == 1: # Long setup active
                if lower_bound <= self.data.Low[-1] <= upper_bound:
                    is_bearish_prev = self.data.Close[-2] < self.data.Open[-2]
                    is_bullish_curr = self.data.Close[-1] > self.data.Open[-1]
                    is_engulfing = self.data.Open[-1] <= self.data.Close[-2] and self.data.Close[-1] >= self.data.Open[-2]

                    if is_bearish_prev and is_bullish_curr and is_engulfing:
                        entry_price = self.data.Close[-1]
                        stop_loss = self.data.Low[-1] * (1 - self.sl_buffer)
                        take_profit = self.data.Low[-1] + (self.point_C - self.data.Low[-1]) * self.tp_extension_level

                        if stop_loss < entry_price and take_profit > entry_price:
                            rr = (take_profit - entry_price) / (entry_price - stop_loss)
                            if rr >= self.min_rr:
                                self.buy(sl=stop_loss, tp=take_profit)
                        self.reset_state() # Reset after attempting a trade

if __name__ == '__main__':
    import os

    # Define the path to the data
    data_path = 'data/crypto/BTC-USD-15m.csv'

    # Load the data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Clean up column names (strip whitespace and capitalize)
    data.columns = [c.strip().title() for c in data.columns]
    # Drop the unnamed column if it exists
    if 'Unnamed: 6' in data.columns:
        data = data.drop(columns=['Unnamed: 6'])

    # Initialize the backtest
    bt = Backtest(data, FibonacciSymmetrySetupStrategy, cash=100_000, commission=.002)

    # Run the backtest
    stats = bt.run()
    print(stats)

    # --- Sanitize and save results ---
    # Helper function to sanitize stats
    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (np.integer, np.int64)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sanitized[key] = float(value) if not np.isnan(value) else None
            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif isinstance(value, pd.Series):
                # For this case, we might just want to summarize or ignore
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

    # Sanitize the main stats object and the trades DataFrame
    results_dict = sanitize_stats(stats)
    if '_trades' in stats and not stats['_trades'].empty:
        results_dict['_trades'] = stats['_trades'].to_dict('records')
        for i, trade in enumerate(results_dict['_trades']):
            results_dict['_trades'][i] = sanitize_stats(trade)
    else:
        results_dict['_trades'] = []

    # Remove specific keys that are not JSON serializable or not needed
    results_dict.pop('_strategy', None)
    results_dict.pop('_equity_curve', None)
    results_dict.pop('_trades_plot', None) # If it exists

    # Final result structure for JSON
    final_result = {
        'strategy_name': 'fibonacci_symmetry_setup',
        'return': results_dict.get('Return [%]'),
        'sharpe': results_dict.get('Sharpe Ratio'),
        'max_drawdown': results_dict.get('Max. Drawdown [%]'),
        'win_rate': results_dict.get('Win Rate [%]'),
        'total_trades': results_dict.get('# Trades', 0)
    }

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Save results to JSON
    with open('results/temp_result.json', 'w') as f:
        json.dump(final_result, f, indent=4)
        f.write('\n') # Add a newline for POSIX compliance

    print("\nResults saved to results/temp_result.json")

    # Generate the plot
    try:
        plot_filename = 'results/fibonacci_symmetry_setup_plot.html'
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
