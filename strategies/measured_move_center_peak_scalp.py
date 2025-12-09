
import json
import os
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

# --- Data Generation and Pre-processing ---

def generate_synthetic_data(num_days=10):
    """
    Generates synthetic 1-minute data with a clear "measured move" pattern.

    The pattern consists of:
    1. A sharp drop (H1 to L1).
    2. A retracement to the 50% Fibonacci level of the drop.
    3. A reversal at that level, forming a smaller peak (H2).
    4. A final drop to a "measured move" target.
    """
    rng = np.random.default_rng(42)
    periods_per_day = 24 * 60
    total_periods = num_days * periods_per_day

    # Base sinusoidal wave for some realistic movement
    t = np.arange(total_periods)
    price = 1000 + 5 * np.sin(t * 0.01) + 2 * np.sin(t * 0.1)

    # Inject the specific patterns
    for day in range(num_days):
        start_idx = day * periods_per_day + 300  # Start pattern 5 hours into the day

        # 1. H1-L1 Drop (lasts ~4 hours)
        h1_idx = start_idx
        l1_idx = h1_idx + 240
        h1_price = price[h1_idx] + 50  # Make H1 a clear peak
        l1_price = h1_price - 100
        price[h1_idx:l1_idx] = np.linspace(h1_price, l1_price, l1_idx - h1_idx)

        # 2. Retracement to 50% AOI
        aoi_idx = l1_idx + 120 # Retracement takes 2 hours
        aoi_price = l1_price + (h1_price - l1_price) * 0.5
        price[l1_idx:aoi_idx] = np.linspace(l1_price, aoi_price, aoi_idx - l1_idx)

        # 3. H2 Reversal Peak (lasts ~30 mins)
        h2_idx = aoi_idx + 15
        h2_price = aoi_price + 5 # Small pop above AOI
        price[aoi_idx:h2_idx] = np.linspace(aoi_price, h2_price, h2_idx - aoi_idx)

        # 4. Final Drop
        target_idx = h2_idx + 120 # Drop takes 2 hours
        target_price = h2_price - (h2_price - l1_price) * 0.5
        price[h2_idx:target_idx] = np.linspace(h2_price, target_price, target_idx - h2_idx)

    # Add noise
    price += rng.normal(0, 0.1, total_periods)

    # Create DataFrame
    index = pd.date_range(start='2023-01-01', periods=total_periods, freq='1min')
    df = pd.DataFrame(price, index=index, columns=['Close'])

    # Generate OHLC data
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0, 0.5, total_periods)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0, 0.5, total_periods)
    df['Volume'] = rng.integers(100, 1000, total_periods)

    return df

# --- Strategy Definition ---

class MeasuredMoveCenterPeakScalpStrategy(Strategy):

    major_peak_lookback = 240
    minor_peak_lookback = 15
    risk_reward_ratio = 5.0 # Target 5:1 risk/reward

    def init(self):
        """
        Initialize the strategy's state machine.
        """
        # Note: Session/Daily High/Low indicators were omitted as the core logic
        # provided did not require them for its entry/exit decisions.
        # The complex "Bag Flip" contingency was also omitted due to ambiguity.

        # Convert params to int for safety
        self.major_lookback = int(self.major_peak_lookback)
        self.minor_lookback = int(self.minor_peak_lookback)

        # State machine states
        self.STATE = "SEARCHING_H1"
        self.h1_price = None
        self.h1_idx = -1
        self.l1_price = None
        self.l1_idx = -1
        self.aoi_level = None

    def next(self):
        """
        Execute the state machine logic on each bar.
        """
        current_idx = len(self.data.Close) - 1

        # --- State: SEARCHING_H1 ---
        # Look for a major peak in the last `major_lookback` bars.
        if self.STATE == "SEARCHING_H1":
            if current_idx < self.major_lookback:
                return

            window = self.data.High[-self.major_lookback:]
            peak_idx = window.argmax()

            # A peak is confirmed if it's in the center of the window
            if peak_idx == self.major_lookback // 2:
                self.h1_price = window[peak_idx]
                self.h1_idx = current_idx - (self.major_lookback // 2)
                self.STATE = "SEARCHING_L1"

        # --- State: SEARCHING_L1 ---
        # After finding H1, look for a subsequent major trough.
        elif self.STATE == "SEARCHING_L1":
            # If price makes a new high, invalidate H1 and reset
            if self.data.High[-1] > self.h1_price:
                self._reset_state()
                return

            window = self.data.Low[self.h1_idx:current_idx]
            if len(window) < self.major_lookback:
                return

            trough_idx = window.argmin()

            # A trough is confirmed if it's not at the very edge of the window
            if 0 < trough_idx < len(window) - 1:
                self.l1_price = window[trough_idx]
                self.l1_idx = self.h1_idx + trough_idx
                self.aoi_level = self.l1_price + (self.h1_price - self.l1_price) * 0.5
                self.STATE = "IN_AOI_SEARCH"

        # --- State: IN_AOI_SEARCH ---
        # Monitor for price to enter the 50% retracement Area of Interest.
        elif self.STATE == "IN_AOI_SEARCH":
            # If price makes a new low, invalidate L1 and reset
            if self.data.Low[-1] < self.l1_price:
                self._reset_state()
                return

            # If price hits 61.8% fib, invalidate
            if self.data.High[-1] > self.l1_price + (self.h1_price - self.l1_price) * 0.618:
                 self._reset_state()
                 return

            # Check for premature rejection at 38.2%
            fib_382_level = self.l1_price + (self.h1_price - self.l1_price) * 0.382
            if self.data.High[-1] < self.aoi_level and self.data.High[-1] > fib_382_level:
                window = self.data.High[-self.minor_lookback:]
                if len(window) == self.minor_lookback and window.argmax() == self.minor_lookback // 2:
                    self._reset_state() # Premature peak found, invalidate
                    return

            # Check for entry into the AOI
            if self.data.High[-1] >= self.aoi_level:
                self.STATE = "SEARCHING_H2"

        # --- State: SEARCHING_H2 ---
        # Once in the AOI, look for a minor reversal peak (H2).
        elif self.STATE == "SEARCHING_H2":
            if self.data.Low[-1] < self.l1_price: # Invalidate if we break the low
                self._reset_state()
                return

            if current_idx - self.l1_idx < self.minor_lookback:
                return

            window = self.data.High[-self.minor_lookback:]
            peak_idx = window.argmax()

            # Confirm H2 peak if it's in the center of the minor window
            if peak_idx == self.minor_lookback // 2 and not self.position:
                h2_price = window[peak_idx]
                stop_loss = h2_price * 1.001 # SL just above H2
                take_profit = h2_price - (h2_price - self.l1_price) * 0.5

                risk = stop_loss - self.data.Close[-1]
                reward = self.data.Close[-1] - take_profit

                if risk > 0 and reward / risk >= self.risk_reward_ratio:
                    self.sell(sl=stop_loss, tp=take_profit)

                # Regardless of trade, reset state to look for new pattern
                self._reset_state()

    def _reset_state(self):
        """Helper to reset the state machine."""
        self.STATE = "SEARCHING_H1"
        self.h1_price = None
        self.h1_idx = -1
        self.l1_price = None
        self.l1_idx = -1
        self.aoi_level = None


# --- Backtesting and Optimization ---

if __name__ == '__main__':

    # Generate data
    data = generate_synthetic_data()

    # Initialize Backtest
    bt = Backtest(data, MeasuredMoveCenterPeakScalpStrategy, cash=100_000, commission=.002)

    # Run optimization
    stats = bt.optimize(
        major_peak_lookback=range(180, 301, 30),
        minor_peak_lookback=range(10, 21, 5),
        risk_reward_ratio=[3, 4, 5],
        maximize='Sharpe Ratio',
        constraint=lambda p: p.major_peak_lookback > p.minor_peak_lookback * 5
    )

    print("Best stats:", stats)

    # --- Output Results ---
    os.makedirs('results', exist_ok=True)

    # Get the best stats series, handling the case of no trades
    final_stats = stats

    # Safely get metrics, providing a default value of 0 if the key is missing or NaN
    return_pct = final_stats.get('Return [%]', 0.0)
    sharpe = final_stats.get('Sharpe Ratio', 0.0)
    max_drawdown = final_stats.get('Max. Drawdown [%]', 0.0)
    win_rate = final_stats.get('Win Rate [%]', 0.0)
    total_trades = final_stats.get('# Trades', 0)

    # Save results to JSON
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'measured_move_center_peak_scalp',
            'return': float(return_pct),
            'sharpe': float(sharpe) if not np.isnan(sharpe) else 0.0,
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': int(total_trades)
        }, f, indent=2)

    print("Results saved to results/temp_result.json")

    # Generate and save the plot of the best run
    bt.plot(filename="results/measured_move_center_peak_scalp", open_browser=False)
    print("Plot saved to results/measured_move_center_peak_scalp.html")
