from backtesting import Backtest, Strategy
from backtesting.test import GOOG
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks

def preprocess_data(data, peak_distance=200): # Increased distance for HTF simulation
    """
    Identifies swing highs and lows using scipy.signal.find_peaks.
    """
    # Find peaks (swing highs)
    high_peaks_indices, _ = find_peaks(data.High, distance=peak_distance)
    data['swing_high'] = False
    data.iloc[high_peaks_indices, data.columns.get_loc('swing_high')] = True

    # Find troughs (swing lows)
    low_peaks_indices, _ = find_peaks(-data.Low, distance=peak_distance)
    data['swing_low'] = False
    data.iloc[low_peaks_indices, data.columns.get_loc('swing_low')] = True

    return data

class FibonacciCenterPeakScalpStrategy(Strategy):
    """
    Implements the Fibonacci Center Peak Scalp trading strategy.
    """
    # Add optimizable parameters
    ema_period = 200
    rr_ratio = 5
    ema_aoi_tolerance = 0.01 # 1% tolerance for EMA alignment
    fib_tolerance = 0.05 # 5% tolerance for Fib level

    def init(self):
        """
        Initializes the strategy.
        """
        # Indicators
        self.ema = self.I(lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean(), self.data.Close, self.ema_period)
        # State management
        self.p1_high = None
        self.p1_low = None
        self.p2_high = None
        self.setup_active = False

        # Pre-processed indicators
        self.swing_highs = self.I(lambda x: x, self.data.df['swing_high'], name='swing_high')
        self.swing_lows = self.I(lambda x: x, self.data.df['swing_low'], name='swing_low')

    def next(self):
        """
        Defines the logic for the next trading iteration.
        """
        # If a position is open, we wait for it to close.
        if self.position:
            return

        # If a setup was active but we no longer have a position, it means the trade just closed.
        if self.setup_active:
            # Check if the last closed trade was a profitable short (implying TP hit for the bag flip)
            if len(self.closed_trades) > 0:
                last_trade = self.closed_trades[-1]
                if last_trade.is_short and last_trade.pl > 0:
                    # Bag Flip: Immediately open a LONG position with managed risk
                    entry_price = self.data.Close[-1]
                    # Place SL below the recent swing low that was the TP for the short
                    stop_loss = self.p1_low * 0.999
                    # Target the peak of the previous setup
                    take_profit = self.p2_high

                    if entry_price > stop_loss and entry_price < take_profit:
                        self.buy(size=last_trade.size, sl=stop_loss, tp=take_profit)

            self._reset_state()
            return

        # --- PATTERN RECOGNITION AND ENTRY ---

        # Step 1: Find P1 High
        if self.swing_highs[-1]:
            self.p1_high = self.data.High[-1]
            self.p1_low = None  # Reset to find the next low

        # Step 2: Find subsequent P1 Low
        if self.p1_high and not self.p1_low and self.swing_lows[-1]:
            if self.data.Low[-1] < self.p1_high:
                self.p1_low = self.data.Low[-1]
            else: # A new high was made, reset P1
                self.p1_high = self.data.High[-1]

        # Step 3: Find P2 High after retracement and enter trade
        if self.p1_high and self.p1_low and not self.p2_high:
            fib_level_38 = self.p1_low + 0.382 * (self.p1_high - self.p1_low)
            fib_level_50 = self.p1_low + 0.5 * (self.p1_high - self.p1_low)
            fib_level_61 = self.p1_low + 0.618 * (self.p1_high - self.p1_low)

            if self.swing_highs[-1]:
                self.p2_high = self.data.High[-1]

                # Fibonacci Level Confirmation
                if not (fib_level_50 * (1 - self.fib_tolerance) <= self.p2_high <= fib_level_50 * (1 + self.fib_tolerance)):
                    self._reset_state() # P2 peak is not at the 50% fib level, invalidate
                    return

                # EMA Confirmation
                ema_value_at_p2 = self.ema[-1]
                price_at_p2 = self.p2_high
                if not (abs(price_at_p2 - ema_value_at_p2) / price_at_p2 <= self.ema_aoi_tolerance):
                    self._reset_state() # EMA not aligned, invalidate
                    return

                # --- ENTRY LOGIC ---
                entry_price = self.data.Close[-1]
                stop_loss = self.p2_high * 1.001
                take_profit = self.p1_low + 0.5 * (self.p2_high - self.p1_low)

                if entry_price >= stop_loss or entry_price <= take_profit:
                    self._reset_state()
                    return

                risk = abs(entry_price - stop_loss)
                reward = abs(entry_price - take_profit)

                if risk > 0 and reward / risk >= self.rr_ratio:
                    size = (self.equity * 0.02) / risk
                    self.sell(size=size, sl=stop_loss, tp=take_profit)
                    self.setup_active = True

    def _reset_state(self):
        """Resets the state of the pattern recognition."""
        self.p1_high = None
        self.p1_low = None
        self.p2_high = None
        self.setup_active = False

def generate_synthetic_data():
    """
    Generates synthetic data with a clear P1 drop -> retracement -> P2 peak pattern.
    """
    n = 2000 # More data points for finer resolution
    index = pd.to_datetime(pd.date_range('2022-01-01', periods=n, freq='1min'))
    data = pd.DataFrame(index=index)

    # Base price with some noise
    price = 100 + np.cumsum(np.random.randn(n) * 0.1)
    data['Open'] = price
    data['High'] = price + np.random.uniform(0, 0.5, n)
    data['Low'] = price - np.random.uniform(0, 0.5, n)
    data['Close'] = price + np.random.randn(n) * 0.2
    data['Volume'] = np.random.randint(100, 1000, n)

    # Create the P1 drop
    p1_high_idx = 200
    p1_low_idx = 400
    data.loc[data.index[p1_high_idx:p1_low_idx], 'Close'] -= np.linspace(0, 10, p1_low_idx - p1_high_idx)

    # Create the retracement to P2 peak
    p2_peak_idx = 500
    p1_high = data['Close'].iloc[p1_high_idx]
    p1_low = data['Close'].iloc[p1_low_idx]
    retracement_target = p1_low + 0.5 * (p1_high - p1_low)

    # Linearly move price to the retracement target
    current_price = data['Close'].iloc[p1_low_idx]
    price_move = np.linspace(current_price, retracement_target, p2_peak_idx - p1_low_idx)
    data.loc[data.index[p1_low_idx:p2_peak_idx], 'Close'] = price_move

    # Create the subsequent drop
    data.loc[data.index[p2_peak_idx:], 'Close'] -= np.linspace(0, 5, n - p2_peak_idx)

    # Ensure OHLC consistency
    data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 0.2, n)
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 0.2, n)

    return data

if __name__ == '__main__':
    # Load or generate data
    data = generate_synthetic_data()
    # Pass a larger peak distance to simulate HTF analysis
    data = preprocess_data(data, peak_distance=100)

    # Run backtest
    bt = Backtest(data, FibonacciCenterPeakScalpStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(
        ema_period=range(50, 250, 50),
        rr_ratio=range(3, 8, 1),
        ema_aoi_tolerance=[i * 0.005 for i in range(1, 5)], # 0.5% to 2.0%
        fib_tolerance=[i * 0.01 for i in range(1, 6)], # 1% to 5%
        maximize='Sharpe Ratio'
    )

    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'fibonacci_center_peak_scalp',
            'return': float(stats.get('Return [%]', 0)),
            'sharpe': float(stats.get('Sharpe Ratio', 0)),
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)),
            'win_rate': float(stats.get('Win Rate [%]', 0)),
            'total_trades': int(stats.get('# Trades', 0))
        }, f, indent=2)

    # Generate plot
    try:
        bt.plot()
    except Exception as e:
        print(f"Error plotting: {e}")
