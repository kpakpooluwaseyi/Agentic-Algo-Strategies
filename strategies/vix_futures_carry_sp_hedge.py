
import json
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import os

def generate_synthetic_vix_data(n_points=2000):
    """
    Generates a synthetic dataset mimicking VIX spot and futures prices.
    This creates scenarios of contango and backwardation needed to test the strategy.
    """
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='B') # Use business days

    # 1. Generate a base VIX spot price series (random walk)
    vix_spot = 15 + np.random.randn(n_points).cumsum() * 0.5
    vix_spot = np.clip(vix_spot, 5, 50) # Keep VIX in a realistic range

    # 2. Simulate days to settlement (cycles from 25 down to 10)
    days_to_settlement = np.zeros(n_points)
    current_days = 25
    for i in range(n_points):
        days_to_settlement[i] = current_days
        current_days -= 1
        if current_days < 10:
            current_days = 25

    # 3. Generate VIX futures price based on spot
    basis = np.zeros(n_points)
    # Most of the time, slight contango (not enough to trigger a trade)
    basis.fill(0.5)
    # Add random shocks to create trading opportunities
    i = 0
    while i < n_points:
        if np.random.rand() < 0.05: # 5% chance of a significant event
            event_type = np.random.choice(['contango', 'backwardation'])
            event_duration = np.random.randint(8, 15)
            # Ensure the basis is large enough to create a clear signal
            min_basis = 0.11 * 25 # Calculate basis for max days to guarantee signal
            basis_value = np.random.uniform(min_basis, min_basis + 3)

            if event_type == 'backwardation':
                basis_value *= -1

            end_idx = min(i + event_duration, n_points)
            basis[i:end_idx] = basis_value
            i += event_duration
        else:
            i += 1

    vix_futures = vix_spot + basis

    # 4. Create OHLC data based on the VIX futures price
    close = vix_futures
    open_price = close - np.random.uniform(-0.5, 0.5, n_points)
    high = np.maximum(close, open_price) + np.random.uniform(0, 0.5, n_points)
    low = np.minimum(close, open_price) - np.random.uniform(0, 0.5, n_points)
    volume = np.random.randint(1000, 5000, n_points)

    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
        'Vix_spot': vix_spot,
        'Vix_futures': vix_futures,
        'Days_to_settlement': days_to_settlement
    }, index=dates)

    return df

class VixFuturesCarrySpHedge(Strategy):
    daily_roll_threshold = 0.10

    def init(self):
        # Data columns are accessed directly in next() via self.data
        pass

    def next(self):
        vix_spot = self.data.Vix_spot[-1]
        vix_futures = self.data.Vix_futures[-1]
        days = self.data.Days_to_settlement[-1]

        if days < 10:
            if self.position:
                self.position.close()
            return

        if days > 0:
            daily_roll = (vix_futures - vix_spot) / days
        else:
            daily_roll = 0

        if self.position:
            if -self.daily_roll_threshold <= daily_roll <= self.daily_roll_threshold:
                self.position.close()

        if not self.position:
            if daily_roll > self.daily_roll_threshold:
                self.sell(size=100)
            elif daily_roll < -self.daily_roll_threshold:
                self.buy(size=100)

def sanitize_stats(stats):
    sanitized = {}
    stats_dict = stats.to_dict() if isinstance(stats, pd.Series) else stats
    for key, value in stats_dict.items():
        if key.startswith('_') or isinstance(value, (pd.DataFrame, pd.Series, type(pd.NA))): continue
        if isinstance(value, (pd.Timestamp, pd.Timedelta)): sanitized[key] = str(value)
        elif isinstance(value, (np.int64, np.float64)): sanitized[key] = value.item()
        elif pd.isna(value): sanitized[key] = None
        else: sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    data = generate_synthetic_vix_data(n_points=2000)

    bt = Backtest(data, VixFuturesCarrySpHedge, cash=100000, commission=.001)
    stats = bt.run()
    print(stats)

    os.makedirs('results', exist_ok=T)
    sanitized_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    plot_filename = 'results/vix_futures_carry_sp_hedge.html'
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
