import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import json
import os
import numpy as np

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it's JSON-serializable.
    Converts specific numpy types and pandas objects to native Python types.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, Strategy):
            continue
        if isinstance(value, (pd.DataFrame, pd.Series)):
            continue
        elif isinstance(value, (np.integer, np.int_)):
            sanitized[key] = int(value)
        elif isinstance(value, np.floating):
            sanitized[key] = float(value)
        elif isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized


class SmaCross(Strategy):
    # Use standard long-term SMA periods
    fast_len = 50
    slow_len = 200

    # Define SL and TP as a percentage of the entry price
    sl_pct = 0.05  # 5% stop loss
    tp_pct = 0.10  # 10% take profit


    def init(self):
        # Convert data to pandas Series for pandas_ta
        self.close_series = pd.Series(self.data.Close, index=self.data.index)

        # Precompute the two moving averages
        self.sma1 = self.I(ta.sma, self.close_series, length=self.fast_len)
        self.sma2 = self.I(ta.sma, self.close_series, length=self.slow_len)

    def next(self):
        entry_price = self.data.Close[-1]

        # Calculate SL and TP levels
        sl_long = entry_price * (1 - self.sl_pct)
        tp_long = entry_price * (1 + self.tp_pct)
        sl_short = entry_price * (1 + self.sl_pct)
        tp_short = entry_price * (1 - self.tp_pct)

        # If the fast MA crosses above the slow MA, go long
        if crossover(self.sma1, self.sma2):
            if not self.position and sl_long < entry_price:
                self.buy(size=0.1, sl=sl_long, tp=tp_long)

        # If the fast MA crosses below the slow MA, go short
        elif crossover(self.sma2, self.sma1):
            if not self.position and sl_short > entry_price:
                self.sell(size=0.1, sl=sl_short, tp=tp_short)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Sanitize column names
    data.columns = [c.strip().title() for c in data.columns]

    # Increase cash to avoid margin issues with high-priced assets
    bt = Backtest(data, SmaCross, cash=1_000_000, commission=.002)
    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    sanitized_stats = sanitize_stats(stats)
    sanitized_stats['strategy_name'] = 'strategy_d3707c51153b'

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    try:
        plot_filename = 'results/strategy_d3707c51153b.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
