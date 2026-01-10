
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest

class MinimalStrategy(Strategy):
    def init(self):
        pass
    def next(self):
        pass

if __name__ == '__main__':
    data = {
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(200, 300, 100),
        'Low': np.random.uniform(50, 100, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.uniform(1000, 5000, 100)
    }
    df = pd.DataFrame(data, index=pd.to_datetime(pd.date_range('2022-01-01', periods=100, freq='15T')))

    print("--- Minimal Test DataFrame Columns ---")
    print(df.columns)
    print("------------------------------------")

    bt = Backtest(df, MinimalStrategy, cash=100_000)
    stats = bt.run()
    print("--- Minimal Test Successful ---")
    print(stats)
