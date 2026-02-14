"""
Minimal Backtesting.py Sanity Check
"""
import sys
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.test import SMA

class MinimalSmaCross(Strategy):
    def init(self):
        print("Initializing SMA Strategy...")
        self.sma50 = self.I(SMA, self.data.Close, 50)
        self.sma200 = self.I(SMA, self.data.Close, 200)

    def next(self):
        if self.sma50 > self.sma200:
            self.buy()
        else:
            self.sell()

if __name__ == '__main__':
    print("--- Starting Sanity Check ---")
    data_path = 'data/BTC-USD-15m.csv'

    try:
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.columns = [col.capitalize() for col in df.columns]
        print(f"Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    if df.empty:
        print("Dataframe is empty.")
        sys.exit(1)

    print("Initializing Backtest...")
    bt = Backtest(df, MinimalSmaCross, cash=100_000)
    print("Running Backtest...")
    stats = bt.run()
    print("Backtest finished.")

    print("--- Sanity Check Results ---")
    print(stats)
    print("--- Script Finished ---")
