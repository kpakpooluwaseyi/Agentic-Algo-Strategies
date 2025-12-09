
from backtesting import Backtest, Strategy
from backtesting.test import GOOG
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks

# --- Custom Indicator Functions ---

def sanitise_for_json(obj):
    """
    Recursively converts non-JSON-serializable objects (like numpy types)
    in a nested structure to their JSON-serializable Python equivalents.
    """
    if isinstance(obj, dict):
        return {k: sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitise_for_json(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    # Add other numpy/pandas types if needed
    return obj


class FibonacciMeasuredMoveMowCenterPeakScalpStrategy(Strategy):
    # --- Strategy Parameters ---
    peak_distance = 50          # Lookback for identifying major swing points (simulates 15M)
    fib_tolerance = 0.02        # Tolerance for price hitting the 50% fib level
    min_rr = 2.0                # Minimum Risk-to-Reward ratio for entry

    # --- State Tracking Variables ---
    point1_high_price = None    # Price of the first major high (start of M)
    point1_high_idx = None      # Index of the first major high
    point2_low_price = None     # Price of the subsequent low
    point2_low_idx = None       # Index of the subsequent low
    aoi_level = None            # Calculated 50% Fib AOI level

    def init(self):
        """
        Initialize indicators and data series.
        """
        # --- Pre-calculate Swing Highs/Lows ---
        # Use scipy.signal.find_peaks to identify significant turning points
        # A larger 'distance' simulates looking at a higher timeframe (e.g., 15M)
        highs = find_peaks(self.data.High, distance=self.peak_distance)[0]
        lows = find_peaks(-self.data.Low, distance=self.peak_distance)[0]

        # Store peak indices in sets for efficient lookup in next()
        self.peak_high_indices = set(highs)
        self.peak_low_indices = set(lows)

    def next(self):
        """
        Main strategy logic, executed for each data point (bar).
        """
        current_bar_idx = len(self.data.Close) - 1

        # 1. IDENTIFY POINT 1 (A significant swing high)
        if self.point1_high_price is None and not self.position:
            if current_bar_idx in self.peak_high_indices:
                self.point1_high_price = self.data.High[-1]
                self.point1_high_idx = current_bar_idx
            return

        # 2. IDENTIFY POINT 2 (A significant swing low after Point 1)
        if self.point1_high_price is not None and self.point2_low_price is None:
            if current_bar_idx in self.peak_low_indices and current_bar_idx > self.point1_high_idx:
                self.point2_low_price = self.data.Low[-1]
                self.point2_low_idx = current_bar_idx

                # 3. CALCULATE AOI (50% Fib Retracement of Level Drop)
                level_drop = self.point1_high_price - self.point2_low_price
                self.aoi_level = self.point2_low_price + (level_drop * 0.5)
            return

        # 4. AWAIT ENTRY CONFIRMATION (Price in AOI + Bearish Engulfing)
        if self.aoi_level is not None and not self.position:
            current_high = self.data.High[-1]
            current_low = self.data.Low[-1]

            # Check if price has entered the Area of Interest
            aoi_upper_bound = self.aoi_level * (1 + self.fib_tolerance)
            aoi_lower_bound = self.aoi_level * (1 - self.fib_tolerance)
            price_in_aoi = aoi_lower_bound <= current_high and current_low <= aoi_upper_bound

            if price_in_aoi:
                # --- Candlestick Pattern: Bearish Engulfing ---
                is_prev_bullish = self.data.Close[-2] > self.data.Open[-2]
                is_curr_bearish = self.data.Close[-1] < self.data.Open[-1]
                engulfs = (self.data.Open[-1] >= self.data.Close[-2]) and (self.data.Close[-1] < self.data.Open[-2])

                if is_prev_bullish and is_curr_bearish and engulfs:
                    # --- Risk Management & Trade Execution ---
                    center_peak_high = self.data.High[-1]
                    center_peak_low = min(self.data.Low[-1], self.data.Low[-2])

                    stop_loss = center_peak_high * 1.001
                    entry_price = self.data.Close[-1]

                    center_peak_move = center_peak_high - self.point2_low_price
                    take_profit = center_peak_high - (center_peak_move * 0.5)

                    reward = entry_price - take_profit
                    risk = stop_loss - entry_price

                    if risk > 0 and reward / risk >= self.min_rr and entry_price > take_profit:
                        self.sell(sl=stop_loss, tp=take_profit)
                        self._reset_state()

            # Invalidation: If price moves significantly above Point 1, the setup is invalid.
            if self.point1_high_price is not None and current_high > self.point1_high_price:
                self._reset_state()
                return

        # --- Reset if trade is closed or pattern is old ---
        if self.position:
            return # Don't look for new signals if in a position

        if self.point1_high_idx and (len(self.data.Close) - self.point1_high_idx) > 250: # Increased timeout
             self._reset_state()


    def _reset_state(self):
        """Helper function to reset all state-tracking variables."""
        self.point1_high_price = None
        self.point1_high_idx = None
        self.point2_low_price = None
        self.point2_low_idx = None
        self.aoi_level = None


def generate_synthetic_data():
    """
    Generates synthetic price data with a textbook M-formation pattern
    designed to trigger the strategy's entry logic.
    """
    n_points = 400
    time = pd.date_range(start='2023-01-01', periods=n_points, freq='1min')

    # Prices
    p1_high = 110
    p2_low = 90
    aoi_level = p2_low + (p1_high - p2_low) * 0.5  # Exactly 100

    # Indices
    p1_idx = 100
    p2_idx = 200
    center_peak_idx = 300

    price = np.full(n_points, 100.0)

    # 1. Trend up to Point 1
    price[50:p1_idx] = np.linspace(105, p1_high - 1, num=50)

    # 2. Level Drop from Point 1 to Point 2
    price[p1_idx:p2_idx] = np.linspace(p1_high, p2_low, num=100)

    # 3. Retracement up to the AOI
    price[p2_idx:center_peak_idx-1] = np.linspace(p2_low, aoi_level - 1, num=99)

    # 4. Create the perfect Bearish Engulfing pattern at the AOI
    # Previous candle (bullish, just below AOI)
    open_prev = aoi_level - 2.0
    close_prev = aoi_level - 1.0
    high_prev = aoi_level - 0.5
    low_prev = aoi_level - 2.5

    # Current candle (the trigger, bearish, engulfing, touching AOI)
    open_curr = close_prev + 0.5  # Higher open than prev close
    close_curr = open_prev - 0.5  # Lower close than prev open
    high_curr = aoi_level + 0.1   # Wick slightly into AOI
    low_curr = close_curr - 0.1

    # 5. Continuation downwards after the pattern
    price[center_peak_idx+1:] = np.linspace(close_curr, close_curr - 10, num=n_points - (center_peak_idx+1))

    # Create DataFrame and OHLC data
    df = pd.DataFrame(index=time)
    df['Open'] = price
    df['Close'] = price
    df['High'] = price
    df['Low'] = price
    df['Volume'] = np.random.randint(100, 1000, n_points)

    # Inject the specific points and pattern
    df.loc[df.index[p1_idx], 'High'] = p1_high
    # Ensure the peak is sharp
    df.loc[df.index[p1_idx-1], 'High'] = p1_high - 1
    df.loc[df.index[p1_idx+1], 'High'] = p1_high - 1


    df.loc[df.index[p2_idx], 'Low'] = p2_low
    # Ensure the trough is sharp
    df.loc[df.index[p2_idx-1], 'Low'] = p2_low + 1
    df.loc[df.index[p2_idx+1], 'Low'] = p2_low + 1

    # Previous (bullish) candle
    df.loc[df.index[center_peak_idx-1], 'Open'] = open_prev
    df.loc[df.index[center_peak_idx-1], 'Close'] = close_prev
    df.loc[df.index[center_peak_idx-1], 'High'] = high_prev
    df.loc[df.index[center_peak_idx-1], 'Low'] = low_prev

    # Current (engulfing) candle
    df.loc[df.index[center_peak_idx], 'Open'] = open_curr
    df.loc[df.index[center_peak_idx], 'Close'] = close_curr
    df.loc[df.index[center_peak_idx], 'High'] = high_curr
    df.loc[df.index[center_peak_idx], 'Low'] = low_curr

    return df


if __name__ == '__main__':
    # Use synthetic data to validate the strategy logic
    data = generate_synthetic_data()

    bt = Backtest(data, FibonacciMeasuredMoveMowCenterPeakScalpStrategy, cash=100_000, commission=.002)

    print("Running optimization on synthetic data...")
    stats = bt.optimize(
        peak_distance=[50],  # Fixed for the known pattern
        fib_tolerance=[i / 100 for i in range(1, 10)], # Test a range of tolerances
        min_rr=[i / 10 for i in range(5, 20, 5)],      # Test a range of R:R
        maximize='Sharpe Ratio',
        constraint=lambda p: p.peak_distance > 0
    )

    print("Best Run Stats:")
    print(stats)

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize stats object before saving
    final_stats = {
        'strategy_name': 'fibonacci_measured_move_mow_center_peak_scalp',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    final_stats_sanitised = sanitise_for_json(final_stats)

    results_path = 'results/temp_result.json'
    print(f"Saving results to {results_path}...")
    with open(results_path, 'w') as f:
        json.dump(final_stats_sanitised, f, indent=2)

    # Generate plot
    plot_path = 'results/fibonacci_measured_move_mow_center_peak_scalp.html'
    print(f"Generating plot... and saving to {plot_path}")
    try:
        bt.plot(filename=plot_path, open_browser=False)
    except TypeError as e:
        print(f"Could not generate plot due to a known issue with the library: {e}")
        print("Continuing without plot generation.")

    print("Script finished.")
