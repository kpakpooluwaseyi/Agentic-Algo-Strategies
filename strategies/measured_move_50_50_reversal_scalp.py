
import json
import os
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import warnings
from scipy.signal import find_peaks

# Suppress FutureWarning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

def find_swings_causal(data, distance):
    """
    Finds swing highs and lows in a causal manner with a lag.
    A peak at index `i-distance` is confirmed at index `i` if it's the
    highest high in the window from `i-2*distance` to `i`. This introduces
    a realistic lag to the signal.
    """
    highs = data.High
    lows = data.Low

    peaks = np.zeros_like(highs, dtype=float)
    troughs = np.zeros_like(lows, dtype=float)

    # Note: This is a simple but slow O(n*d) implementation.
    # For production, a more optimized algorithm (e.g., using a deque) would be better.
    for i in range(2 * distance, len(highs)):
        window_highs = highs[i - 2 * distance : i]
        # A peak is confirmed at the center of the lookback window
        if highs[i - distance] >= np.max(window_highs):
            peaks[i - distance] = 1

        window_lows = lows[i - 2 * distance : i]
        # A trough is confirmed at the center of the lookback window
        if lows[i - distance] <= np.min(window_lows):
            troughs[i - distance] = 1

    return peaks, troughs

class MeasuredMove5050ReversalScalpStrategy(Strategy):
    """
    Implements the Measured Move 50/50 Reversal Scalp strategy.

    This strategy identifies M-formation (short) and W-formation (long) setups.
    It enters on a retracement to a 50% Fibonacci level and targets another
    50% level of the subsequent structure.
    """
    # Optimization parameters
    swing_distance = 20
    fib_tolerance = 0.05
    min_rr = 5

    # State tracking variables
    peak_indices = None
    trough_indices = None

    # M-Formation (Short) state
    m_formation_high = None
    m_formation_low = None
    m_center_peak_high = None

    # W-Formation (Long) state
    w_formation_low = None
    w_formation_high = None
    w_center_trough_low = None

    def init(self):
        """
        Initialize the strategy and pre-calculate indicators or signals.
        """
        # Expose OHLC data as indicators for easier access
        self.open = self.I(lambda x: x, self.data.Open)
        self.high = self.I(lambda x: x, self.data.High)
        self.low = self.I(lambda x: x, self.data.Low)
        self.close = self.I(lambda x: x, self.data.Close)

        # Pre-calculate swing points causally
        peaks, troughs = self.I(find_swings_causal, self.data, self.swing_distance)
        # Convert the indicator arrays to indices for easier lookup
        self.peak_indices = np.where(peaks == 1)[0]
        self.trough_indices = np.where(troughs == 1)[0]

    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        current_index = len(self.data) - 1
        current_price = self.close[-1]

        # --- M-Formation (Short) Logic ---
        if self.position.is_short: return # Don't stack trades

        recent_peaks = self.peak_indices[self.peak_indices < current_index]
        recent_troughs = self.trough_indices[self.trough_indices < current_index]

        if len(recent_peaks) > 1 and len(recent_troughs) > 0:
            last_peak_idx = recent_peaks[-2]

            # Find trough that occurred after the last peak
            following_troughs = recent_troughs[recent_troughs > last_peak_idx]
            if len(following_troughs) > 0:
                last_trough_idx = following_troughs[0]

                # Center peak is the most recent peak
                center_peak_idx = recent_peaks[-1]
                if center_peak_idx > last_trough_idx:
                    m_high = self.high[last_peak_idx]
                    m_low = self.low[last_trough_idx]
                    center_peak_high = self.high[center_peak_idx]

                    fib_50_level = m_high - (m_high - m_low) * 0.5

                    if abs(center_peak_high - fib_50_level) / fib_50_level < self.fib_tolerance:
                        if self._is_bearish_reversal():
                            stop_loss = center_peak_high * 1.001
                            take_profit = center_peak_high - (center_peak_high - m_low) * 0.5
                            risk = abs(current_price - stop_loss)
                            reward = abs(current_price - take_profit)

                            if risk > 0 and reward / risk >= self.min_rr:
                                # Ensure TP/SL are valid before placing order
                                if stop_loss > current_price and take_profit < current_price:
                                    self.sell(sl=stop_loss, tp=take_profit)

        # --- W-Formation (Long) Logic ---
        if self.position.is_long: return # Don't stack trades

        if len(recent_troughs) > 1 and len(recent_peaks) > 0:
            last_trough_idx = recent_troughs[-2]

            following_peaks = recent_peaks[recent_peaks > last_trough_idx]
            if len(following_peaks) > 0:
                last_peak_idx = following_peaks[0]

                center_trough_idx = recent_troughs[-1]
                if center_trough_idx > last_peak_idx:
                    w_low = self.low[last_trough_idx]
                    w_high = self.high[last_peak_idx]
                    center_trough_low = self.low[center_trough_idx]

                    fib_50_level = w_low + (w_high - w_low) * 0.5

                    if abs(center_trough_low - fib_50_level) / fib_50_level < self.fib_tolerance:
                        if self._is_bullish_reversal():
                            stop_loss = center_trough_low * 0.999
                            take_profit = center_trough_low + (w_high - center_trough_low) * 0.5
                            risk = abs(current_price - stop_loss)
                            reward = abs(current_price - take_profit)

                            if risk > 0 and reward / risk >= self.min_rr:
                                # Ensure TP/SL are valid before placing order
                                if stop_loss < current_price and take_profit > current_price:
                                    self.buy(sl=stop_loss, tp=take_profit)

    def _is_bearish_reversal(self):
        """
        Checks for a strong bearish candle.
        """
        if len(self.close) < 1: return False
        return self.close[-1] < self.open[-1]

    def _is_bullish_reversal(self):
        """
        Checks for a strong bullish candle.
        """
        if len(self.close) < 1: return False
        return self.close[-1] > self.open[-1]

def generate_synthetic_data(n_points=2000):
    """
    Generates synthetic data with clear M and W formations.
    """
    np.random.seed(42)
    time = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))
    price = 100 + np.random.randn(n_points).cumsum() * 0.1

    # M-Formation
    # High, then low, then a lower high (center peak), then a lower low
    price[300:350] = np.linspace(price[299], 110, 50) # Peak 1
    price[350:400] = np.linspace(price[349], 100, 50) # Trough 1
    price[400:450] = np.linspace(price[399], 105, 50) # Center Peak (50% retracement)
    price[450:500] = np.linspace(price[449], 95, 50)  # Final Low

    # W-Formation
    # Low, then high, then a higher low (center trough), then a higher high
    price[900:950] = np.linspace(price[899], 90, 50)  # Trough 1
    price[950:1000] = np.linspace(price[949], 100, 50) # Peak 1
    price[1000:1050] = np.linspace(price[999], 95, 50) # Center Trough (50% retracement)
    price[1050:1100] = np.linspace(price[1049], 105, 50)# Final High

    # Add some noise
    noise = np.random.normal(0, 0.1, n_points)
    price += noise

    # Create OHLC
    data = pd.DataFrame(index=time)
    data['Open'] = price
    data['High'] = price + np.abs(np.random.normal(0, 0.2, n_points))
    data['Low'] = price - np.abs(np.random.normal(0, 0.2, n_points))
    data['Close'] = price + np.random.normal(0, 0.1, n_points)
    data['Volume'] = np.random.randint(100, 1000, n_points)

    # Ensure High is the max and Low is the min
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data

if __name__ == '__main__':
    data = generate_synthetic_data()

    # --- Backtest ---
    bt = Backtest(data, MeasuredMove5050ReversalScalpStrategy,
                  cash=100_000, commission=.002)

    # --- Optimization ---
    print("Optimizing strategy...")
    stats = bt.optimize(
        swing_distance=range(10, 40, 5),
        fib_tolerance=[i / 100 for i in range(1, 10)], # 0.01 to 0.09
        min_rr=range(3, 8, 1),
        maximize='Sharpe Ratio'
    )

    print("Best Run Stats:")
    print(stats)

    # --- Results & Plotting ---
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, 'temp_result.json')

    def sanitize_stats(stats):
        """
        Recursively sanitizes stats for JSON serialization.
        Converts pd.Series, np.generic, and pd.Timestamp to native types.
        """
        if isinstance(stats, pd.Series):
            stats = stats.to_dict()

        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (np.integer, np.int64)):
                value = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                value = float(value) if not pd.isna(value) else None
            elif isinstance(value, pd.Timestamp):
                value = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                value = str(value)
            elif isinstance(value, (dict, pd.Series)):
                value = sanitize_stats(value)
            # Skip non-serializable types like DataFrame
            elif isinstance(value, pd.DataFrame):
                continue
            sanitized[key] = value
        return sanitized

    # Sanitize the stats object
    stats_dict = stats.to_dict()
    sanitized_stats_full = sanitize_stats(stats_dict)

    # Extract the specific fields as requested
    output_data = {
        'strategy_name': 'measured_move_50_50_reversal_scalp',
        'return': sanitized_stats_full.get('Return [%]'),
        'sharpe': sanitized_stats_full.get('Sharpe Ratio'),
        'max_drawdown': sanitized_stats_full.get('Max. Drawdown [%]'),
        'win_rate': sanitized_stats_full.get('Win Rate [%]'),
        'total_trades': sanitized_stats_full.get('# Trades')
    }

    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {results_path}")

    # Generate plot
    try:
        plot_path = os.path.join(results_dir, 'measured_move_50_50_reversal_scalp.html')
        bt.plot(filename=plot_path)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
