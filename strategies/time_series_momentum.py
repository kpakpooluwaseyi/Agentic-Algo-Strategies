import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest
import os
import json
import numpy as np

class TimeSeriesMomentum(Strategy):
    """
    Implements the Time Series Momentum (Absolute Momentum) strategy.

    Long Entry: Buys if the past 12-month return is positive.
    Exit: Sells if the past 12-month return is negative.
    """
    # --- Strategy Parameters ---
    lookback_period = 252 # Approximate number of trading days in 12 months
    sl_pct = 0.10 # 10% stop-loss
    tp_pct = 0.20 # 20% take-profit

    def init(self):
        """
        Initialize strategy. The momentum signal is pre-calculated and passed in the data.
        """
        # The pre-calculated momentum signal is accessed directly from the data feed.
        # A value > 0 indicates positive momentum, < 0 indicates negative momentum.
        self.momentum = self.I(lambda x: x, self.data.momentum_signal, name="MomentumSignal")

    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        price = self.data.Close[-1]

        # --- Entry Logic ---
        # If not in a position and momentum is positive, go long.
        if not self.position and self.momentum[-1] > 0:
            sl = price * (1 - self.sl_pct)
            tp = price * (1 + self.tp_pct)
            self.buy(sl=sl, tp=tp)

        # --- Exit Logic ---
        # If in a position and momentum turns negative, close the position.
        elif self.position and self.momentum[-1] <= 0:
            self.position.close()

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    # --- Data Loading and Preprocessing ---
    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Clean and rename columns to Backtesting.py convention
        data.columns = data.columns.str.strip()
        data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    else:
        print(f"Data file not found at {data_path}. Generating synthetic data.")
        n_points = 50000 # ~1 year of 15m data
        index = pd.date_range('2023-01-01', periods=n_points, freq='15min')
        price = 100 + np.log(np.arange(1, n_points + 1)) * 20
        # Create a momentum shift mid-way
        price[n_points//2:] = price[n_points//2 -1] - np.log(np.arange(1, n_points//2 + 1)) * 20

        volume = np.random.uniform(100, 500, n_points)
        data = pd.DataFrame({
            'Open': price, 'High': price + 0.5, 'Low': price - 0.5, 'Close': price, 'Volume': volume
        }, index=index)

    # --- Signal Calculation ---
    # Resample to daily to calculate annual momentum
    daily_close = data['Close'].resample('D').last()

    # Calculate 12-month (252 trading days) percentage change
    # Note: Using pct_change is equivalent to (price / price.shift(period)) - 1
    daily_momentum = daily_close.pct_change(periods=TimeSeriesMomentum.lookback_period)

    # Map daily momentum signal back to the 15m dataframe
    data['momentum_signal'] = daily_momentum.reindex(data.index, method='ffill')

    # Drop rows with NaN momentum signal (the initial lookback period)
    data.dropna(inplace=True)

    # --- Backtesting ---
    bt = Backtest(data, TimeSeriesMomentum, cash=100_000, commission=.002, finalize_trades=True)

    print("Running backtest with default parameters...")
    stats = bt.run()

    print("\nBacktest Stats:")
    print(stats)

    # --- Results and Plotting ---
    os.makedirs('results', exist_ok=True)

    # Sanitize the stats object for JSON output
    # Remove non-serializable objects (like the strategy instance and trades dataframe)
    sanitized_stats = {key: value for key, value in stats.items() if not key.startswith('_')}

    # Convert specific types if necessary (e.g., Timestamps, Timedeltas to strings)
    for key, value in sanitized_stats.items():
        if isinstance(value, pd.Timestamp):
            sanitized_stats[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized_stats[key] = str(value)

    # Save stats to JSON
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)
    print("\nResults saved to results/temp_result.json")

    # Generate plot
    try:
        plot_filename = 'results/time_series_momentum.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
