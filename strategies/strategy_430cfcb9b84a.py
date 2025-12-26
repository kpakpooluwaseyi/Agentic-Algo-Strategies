from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta

def sma(series, n):
    """Wrapper for pandas-ta sma to be used with self.I"""
    return ta.sma(pd.Series(series), length=n).values

class MovingAverageCrossover(Strategy):
    """
    A simple moving average crossover strategy.
    Buy when the fast MA crosses above the slow MA.
    Sell when the fast MA crosses below the slow MA.
    """
    # Default parameters
    fast_ma_period = 10
    slow_ma_period = 20
    sl_pct = 0.05  # 5% stop loss

    def init(self):
        """
        Initialize indicators.
        """
        self.fast_ma = self.I(sma, self.data.Close, self.fast_ma_period)
        self.slow_ma = self.I(sma, self.data.Close, self.slow_ma_period)

    def next(self):
        """
        Define the trading logic for each bar.
        """
        entry_price = self.data.Close[-1]

        # --- LONG ENTRY ---
        if crossover(self.fast_ma, self.slow_ma):
            # Close any open short position
            if self.position.is_short:
                self.position.close()

            # Go long
            if not self.position.is_long:
                sl = entry_price * (1 - self.sl_pct)
                self.buy(sl=sl)

        # --- SHORT ENTRY ---
        elif crossover(self.slow_ma, self.fast_ma):
            # Close any open long position
            if self.position.is_long:
                self.position.close()

            # Go short
            if not self.position.is_short:
                sl = entry_price * (1 + self.sl_pct)
                self.sell(sl=sl)


if __name__ == '__main__':
    import os
    import json
    from backtesting import Backtest
    import pandas as pd

    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    strategy_name = 'strategy_430cfcb9b84a'

    # --- Data Loading ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.strip().title() for c in data.columns] # Sanitize headers

    # --- Backtesting ---
    bt = Backtest(data, MovingAverageCrossover, cash=100_000, commission=.002)
    stats = bt.run()

    # --- Results Saving ---
    os.makedirs('results', exist_ok=True)

    # Convert stats Series to a dictionary for manipulation
    stats_dict = dict(stats)

    # Remove non-serializable items from the dictionary
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    # A simple loop to sanitize the rest of the values
    import numpy as np
    for key, value in list(stats_dict.items()):
        if isinstance(value, (np.integer, np.int_)):
            stats_dict[key] = int(value)
        elif isinstance(value, np.floating):
            stats_dict[key] = float(value)
        elif isinstance(value, pd.Timestamp):
            stats_dict[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            stats_dict[key] = str(value)
        elif pd.isna(value) or value is None:
            stats_dict[key] = None

    # Save stats
    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    # --- Plotting ---
    try:
        plot_filename = f'results/{strategy_name}.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
