"""
Dollar Cost Averaging (DCA) Strategy
=====================================

This strategy implements the Dollar Cost Averaging investment methodology.
It systematically invests a fixed dollar amount at regular intervals,
regardless of the asset's price.
"""

from backtesting import Strategy
from backtesting.lib import FractionalBacktest
import pandas as pd
import numpy as np


class DollarCostAveraging(Strategy):
    """
    Invests a fixed amount of USD at regular intervals.

    This strategy simulates a pure Dollar Cost Averaging approach.

    NOTE ON GUIDELINES: Standard active trading guidelines (e.g., ATR-based
    stops/profits, volume confirmation, multi-timeframe filters) have been
    intentionally omitted. Such rules are philosophically contrary to the
    core DCA principle, which is to invest consistently over time regardless
    of market conditions.

    This implementation requires using the `FractionalBacktest` class to
    handle the fractional position sizing that results from investing a
    fixed dollar amount.

    Parameters:
    - investment_amount_usd: The amount in USD to invest at each interval.
    - interval_days: The number of days between each investment.
    """

    # Optimizable parameters
    investment_amount_usd = 100
    interval_days = 7

    def init(self):
        """
        Initialize the strategy.
        """
        # Dynamically calculate the number of bars per day from the data
        time_diff = np.median(np.diff(self.data.index.values))
        bars_per_day = pd.Timedelta(days=1) / time_diff

        self.interval_bars = self.interval_days * bars_per_day

        # Start investing on the first possible bar
        self.last_investment_bar = -self.interval_bars

    def next(self):
        """
        The main trading logic. Called on each bar.
        """
        current_bar = len(self.data) - 1

        # Check if the defined interval has passed since the last investment
        if current_bar - self.last_investment_bar >= self.interval_bars:
            # We must calculate the size as a fraction of equity for FractionalBacktest
            equity = self.equity
            current_price = self.data.Close[-1]

            # Ensure we don't try to invest more than we have
            if equity > self.investment_amount_usd and current_price > 0:
                size_as_fraction = self.investment_amount_usd / equity

                # Place the buy order using the calculated fraction of equity
                self.buy(size=size_as_fraction)

                # Record the bar of this investment
                self.last_investment_bar = current_bar


# Standalone execution
if __name__ == '__main__':
    # Load data
    try:
        # The user-specified data path
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
        # Sanitize column names (e.g., 'open' -> 'Open')
        df.columns = [col.strip().capitalize() for col in df.columns]
        print("Successfully loaded and sanitized data/BTC-USD-15m.csv")
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found.")
        print("Please ensure you have the correct data file in the 'data' directory.")
        # As a fallback, create synthetic data for demonstration
        print("Generating synthetic data for demonstration purposes...")
        dates = pd.date_range(start='2020-01-01', periods=35040, freq='15min') # 1 year of data
        price = 60000 + np.cumsum(np.random.randn(35040) * 50)
        price = np.maximum(price, 1000) # Ensure price doesn't go negative
        df = pd.DataFrame({
            'Open': price,
            'High': price + np.random.uniform(0, 100, 35040),
            'Low': price - np.random.uniform(0, 100, 35040),
            'Close': price + np.random.randn(35040) * 25,
            'Volume': np.random.uniform(1, 200, 35040)
        }, index=dates)
        df['Close'] = np.maximum(df['Close'], 1) # Ensure close is positive


    # Run the backtest using FractionalBacktest
    # finalize_trades=True will close any open positions at the end of the backtest
    # to include them in the final performance statistics.
    bt = FractionalBacktest(df, DollarCostAveraging, cash=100_000, commission=.002, finalize_trades=True)
    stats = bt.run()

    print("--- Dollar Cost Averaging Strategy ---")
    print(stats)

    # Save results and plot
    import os
    if not os.path.exists('results'):
        os.makedirs('results')

    # Sanitize stats for JSON serialization
    sanitized_stats = {}
    for key, value in stats.items():
        # Skip non-serializable pandas objects and the strategy object itself
        if isinstance(value, (pd.Series, pd.DataFrame)) or key == '_strategy':
            continue
        if isinstance(value, (np.integer, np.floating)):
            sanitized_stats[key] = value.item()
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized_stats[key] = str(value)
        elif pd.isna(value):
            sanitized_stats[key] = None
        else:
            sanitized_stats[key] = value

    import json
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print("\nBacktest stats saved to results/temp_result.json")

    try:
        bt.plot(filename='results/dollar_cost_averaging.html', open_browser=False)
        print("Backtest plot saved to results/dollar_cost_averaging.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
