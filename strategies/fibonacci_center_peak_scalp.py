import numpy as np
import pandas as pd
import json
from backtesting import Backtest, Strategy
import os

def generate_synthetic_data(n_patterns=20, points_per_pattern=100, noise_level=0.05):
    """
    Generates synthetic OHLC data with M-shaped patterns suitable for testing
    the Fibonacci Center Peak Scalp strategy.
    """
    close_prices = []
    base_price = 100

    for _ in range(n_patterns):
        # A: Initial high
        p_a = base_price + np.random.uniform(5, 10)
        # B: Initial low
        p_b = p_a - np.random.uniform(10, 20)
        # C: Retracement high to 50% of A-B drop
        p_c = p_b + 0.5 * (p_a - p_b) + np.random.uniform(-0.5, 0.5)
        # D: Final low, lower than B
        p_d = p_b - np.random.uniform(2, 5)

        segment1 = np.linspace(base_price, p_a, points_per_pattern // 4)
        segment2 = np.linspace(p_a, p_b, points_per_pattern // 4)
        segment3 = np.linspace(p_b, p_c, points_per_pattern // 4)
        segment4 = np.linspace(p_c, p_d, points_per_pattern // 4)

        close_prices.extend(np.concatenate([segment1, segment2, segment3, segment4]))
        base_price = p_d

    close_prices = np.array(close_prices)
    noise = np.random.normal(0, noise_level, len(close_prices))
    close_prices += noise

    df = pd.DataFrame(index=pd.date_range(start='2023-01-01', periods=len(close_prices), freq='15min'))
    df['Close'] = close_prices
    df['Open'] = df['Close'].shift(1).fillna(method='bfill')
    df['High'] = pd.Series([max(o, c) for o, c in zip(df['Open'], df['Close'])], index=df.index) + np.random.uniform(0, noise_level * 2, len(df))
    df['Low'] = pd.Series([min(o, c) for o, c in zip(df['Open'], df['Close'])], index=df.index) - np.random.uniform(0, noise_level * 2, len(df))

    return df[['Open', 'High', 'Low', 'Close']]

def position_size(equity, risk_pct, stop_loss_price, entry_price):
    """Calculates position size based on fixed fractional risk."""
    risk_amount = equity * risk_pct
    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share == 0:
        return 0
    size = risk_amount / risk_per_share
    return max(0, int(size)) # Ensure non-negative integer size

class FibonacciCenterPeakScalpStrategy(Strategy):
    # --- Strategy Parameters ---
    # How many bars to look left and right to confirm a swing point
    swing_confirmation = 5
    # Tolerance for Fibonacci level matching (e.g., 0.05 = 5%)
    fib_tolerance = 0.05

    def init(self):
        # State machine for M-pattern detection
        self.state = 'SCANNING_A' # Initial state
        self.point_a = None
        self.point_b = None
        self.point_c = None

        # Store index of points for time-based checks
        self.idx_a = None
        self.idx_b = None
        self.idx_c = None

    def is_swing_high(self, index_offset):
        """ Checks if a point `index_offset` bars ago is a swing high. """
        if len(self.data.High) < 2 * self.swing_confirmation + 1:
            return False

        window = self.data.High[-(2 * self.swing_confirmation + 1) + index_offset : index_offset if index_offset != 0 else None]
        if len(window) == 0: return False

        center_index = self.swing_confirmation
        return window[center_index] == max(window)

    def is_swing_low(self, index_offset):
        """ Checks if a point `index_offset` bars ago is a swing low. """
        if len(self.data.Low) < 2 * self.swing_confirmation + 1:
            return False

        window = self.data.Low[-(2 * self.swing_confirmation + 1) + index_offset : index_offset if index_offset != 0 else None]
        if len(window) == 0: return False

        center_index = self.swing_confirmation
        return window[center_index] == min(window)

    def next(self):
        # We use a lag `swing_confirmation` to confirm a swing point without lookahead bias
        current_bar_index = len(self.data.Close) - 1
        eval_bar_index = current_bar_index - self.swing_confirmation
        if eval_bar_index < 0:
            return

        # --- State Machine for M-Pattern Detection ---

        # State: SCANNING_A - Looking for the first major peak (Point A)
        if self.state == 'SCANNING_A':
            if self.is_swing_high(0):
                self.point_a = self.data.High[-self.swing_confirmation-1]
                self.idx_a = eval_bar_index
                self.state = 'SCANNING_B'

        # State: SCANNING_B - Looking for a swing low after Point A
        elif self.state == 'SCANNING_B':
            if self.is_swing_low(0):
                self.point_b = self.data.Low[-self.swing_confirmation-1]
                self.idx_b = eval_bar_index
                self.state = 'SCANNING_C'
            # Invalidation: If a new higher high is made, reset and look for a new A
            elif self.is_swing_high(0) and self.data.High[-self.swing_confirmation-1] > self.point_a:
                self.point_a = self.data.High[-self.swing_confirmation-1]
                self.idx_a = eval_bar_index

        # State: SCANNING_C - Looking for a retracement peak after Point B
        elif self.state == 'SCANNING_C':
            if self.is_swing_high(0):
                p_c_candidate = self.data.High[-self.swing_confirmation-1]
                # Validation: Point C must be lower than Point A for an M-pattern
                if p_c_candidate < self.point_a:
                    self.point_c = p_c_candidate
                    self.idx_c = eval_bar_index
                    self.state = 'VALIDATING'
                else: # Invalid M-pattern (higher high), reset
                    self.state = 'SCANNING_A'
            # Invalidation: If a new lower low is made, reset B
            elif self.is_swing_low(0) and self.data.Low[-self.swing_confirmation-1] < self.point_b:
                self.point_b = self.data.Low[-self.swing_confirmation-1]
                self.idx_b = eval_bar_index

        # State: VALIDATING - We have A, B, and C. Check Fibonacci levels.
        if self.state == 'VALIDATING':
            # Calculate Fib 1 from A to B
            fib1_range = self.point_a - self.point_b
            if fib1_range > 0:
                fib1_50 = self.point_a - 0.5 * fib1_range

                # Check if C is within tolerance of the 50% level
                if abs(self.point_c - fib1_50) <= fib1_range * self.fib_tolerance:
                    # SETUP CONFIRMED: Now wait for entry trigger on the current bar
                    # This simulates dropping to 1M and finding a reversal.
                    # A simple trigger is a bearish candle (Close < Open).
                    if self.data.Close[-1] < self.data.Open[-1] and not self.position:
                        entry_price = self.data.Close[-1]

                        # SL is just above peak C
                        sl = self.point_c * 1.002

                        # TP is 50% of Fib 2 (from B to C)
                        fib2_range = self.point_c - self.point_b
                        tp = self.point_c - 0.5 * fib2_range

                        # Ensure SL/TP are valid for a short position
                        if sl > entry_price and tp < entry_price:
                            size = position_size(self.equity, 0.02, sl, entry_price)
                            if size > 0:
                                self.sell(size=size, sl=sl, tp=tp)

            # Whether trade was taken or validation failed, reset the state machine
            self.state = 'SCANNING_A'

if __name__ == '__main__':
    data = generate_synthetic_data(n_patterns=50)

    bt = Backtest(data, FibonacciCenterPeakScalpStrategy, cash=100000, commission=.002)

    print("Optimizing strategy...")
    stats = bt.optimize(
        swing_confirmation=range(3, 15, 2),
        fib_tolerance=[i/100 for i in range(2, 8)], # 0.02 to 0.07
        maximize='Sharpe Ratio',
        constraint=lambda p: p.swing_confirmation > 0
    )

    print("\nBest Run Stats:")
    print(stats)

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    result_data = {
        'strategy_name': 'fibonacci_center_peak_scalp',
        'return': float(stats.get('Return [%]', 0.0)),
        'sharpe': float(stats.get('Sharpe Ratio', 0.0)),
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
        'win_rate': float(stats.get('Win Rate [%]', 0.0)),
        'total_trades': int(stats.get('# Trades', 0))
    }

    if result_data['total_trades'] == 0:
        print("Warning: No trades were executed during the backtest.")

    with open(os.path.join(results_dir, 'temp_result.json'), 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"Results saved to {os.path.join(results_dir, 'temp_result.json')}")

    bt.plot(filename="fibonacci_center_peak_scalp_plot")
