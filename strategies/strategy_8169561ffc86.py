from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it's JSON-serializable.
    Converts specific numpy types and pandas objects to native Python types.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            sanitized[key] = None
        elif isinstance(value, (np.integer, np.int_)):
            sanitized[key] = int(value)
        elif isinstance(value, np.floating): # Use np.floating for general float check
            sanitized[key] = float(value)
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value

    # Remove problematic keys
    if '_strategy' in sanitized:
        del sanitized['_strategy']
    if '_equity_curve' in sanitized:
        del sanitized['_equity_curve']
    if '_trades' in sanitized:
        del sanitized['_trades']

    return sanitized

class EmaCrossStrategy(Strategy):
    """
    A simple moving average crossover strategy.
    """
    fast_ema_period = 20
    slow_ema_period = 50
    sl_pct = 0.05 # 5% stop loss

    def init(self):
        """
        Initialize indicators.
        """
        close_series = pd.Series(self.data.Close)
        self.fast_ema = self.I(lambda x: ta.ema(pd.Series(x), length=self.fast_ema_period), self.data.Close)
        self.slow_ema = self.I(lambda x: ta.ema(pd.Series(x), length=self.slow_ema_period), self.data.Close)

    def next(self):
        """
        Define the trading logic.
        """
        price = self.data.Close[-1]

        if crossover(self.fast_ema, self.slow_ema):
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                stop_loss = price * (1 - self.sl_pct)
                self.buy(sl=stop_loss)

        elif crossover(self.slow_ema, self.fast_ema):
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                stop_loss = price * (1 + self.sl_pct)
                self.sell(sl=stop_loss)


if __name__ == '__main__':
    from backtesting import Backtest

    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Clean up column names
    data.columns = [c.strip() for c in data.columns]
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    bt = Backtest(data, EmaCrossStrategy, cash=100000, commission=.002)

    print("Running single backtest with default parameters...")
    stats = bt.run()

    # Save results
    os.makedirs('results', exist_ok=True)

    # Sanitize stats before saving
    result_stats = {
        'Return [%]': stats.get('Return [%]'),
        'Sharpe Ratio': stats.get('Sharpe Ratio'),
        'Max. Drawdown [%]': stats.get('Max. Drawdown [%]'),
        'Win Rate [%]': stats.get('Win Rate [%]'),
        '# Trades': stats.get('# Trades')
    }

    sanitized_result_stats = sanitize_stats(result_stats)

    result = {
        'strategy_name': 'strategy_8169561ffc86',
        'return': sanitized_result_stats.get('Return [%]'),
        'sharpe': sanitized_result_stats.get('Sharpe Ratio'),
        'max_drawdown': sanitized_result_stats.get('Max. Drawdown [%]'),
        'win_rate': sanitized_result_stats.get('Win Rate [%]'),
        'total_trades': sanitized_result_stats.get('# Trades')
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to results/temp_result.json")
    print(stats)

    try:
        plot_filename = 'results/strategy_8169561ffc86.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
