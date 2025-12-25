# Implementation of the Moving Average Bands Reversal Strategy

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import json
import os
import numpy as np

# Wrapper function to ensure the indicator returns a writable numpy array
def sma_indicator(series, length):
    """Calculates SMA and returns a writable numpy array copy."""
    return ta.sma(pd.Series(series), length=length).values.copy()

class MovingAverageBandsReversalStrategy(Strategy):
    """
    A trend-following strategy that uses moving average bands for reversal signals.
    It is always in the market, either long or short.
    """

    # --- Strategy Parameters ---
    ma_period = 20
    band_pct = 2.0

    def init(self):
        """
        Initialize the strategy's indicators.
        """
        # Calculate the Simple Moving Average (SMA) using the wrapper
        self.sma = self.I(sma_indicator, self.data.Close, length=self.ma_period)

        # Calculate the upper and lower bands as a percentage of the SMA
        self.upper_band = self.I(lambda x, y: x * (1 + y / 100), self.sma, self.band_pct)
        self.lower_band = self.I(lambda x, y: x * (1 - y / 100), self.sma, self.band_pct)

    def next(self):
        """
        Define the trading logic for the next bar.
        """
        price = self.data.Close[-1]

        # If not in the market, go long by default to start the cycle
        if not self.position:
            self.buy()

        # Flip-flop logic: Reverse position when the opposite band is breached
        elif self.position.is_long:
            if crossover(self.lower_band, self.data.Close):
                self.position.close()
                self.sell()

        elif self.position.is_short:
            if crossover(self.data.Close, self.upper_band):
                self.position.close()
                self.buy()

def run_backtest(data_path='data/BTC-USD-15m.csv', strategy=MovingAverageBandsReversalStrategy):
    """
    Runs the backtest for the given strategy and data.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        # As a fallback for testing, create some synthetic data
        print("Generating synthetic data for demonstration.")
        price = pd.Series(np.random.randn(5000).cumsum() + 10000, name='Close')
        price.index = pd.to_datetime(pd.date_range('2020-01-01', periods=len(price), freq='15min'))
        data = pd.DataFrame(price)
        data['Open'] = data['Close'].shift()
        data['High'] = data[['Open', 'Close']].max(axis=1) * 1.02
        data['Low'] = data[['Open', 'Close']].min(axis=1) * 0.98
        data['Volume'] = np.random.randint(100, 1000, size=len(price))
        data.dropna(inplace=True)
    else:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Clean column names: strip whitespace and capitalize
        data.columns = [col.strip().title() for col in data.columns]
        # Drop any unnamed columns that might have been read
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

    # Use FractionalBacktest to allow for fractional position sizes
    from backtesting.lib import FractionalBacktest
    bt = FractionalBacktest(data, strategy, cash=100_000, commission=.002, finalize_trades=True)

    print("Running backtest...")
    stats = bt.run()

    print("Backtest Stats:")
    print(stats)

    # Save results to JSON
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Sanitize stats for JSON serialization
    sanitized_stats = {key: val for key, val in stats.items() if not isinstance(val, (pd.DataFrame, pd.Series))}
    sanitized_stats = {
        'strategy_name': strategy.__name__,
        'return_pct': sanitized_stats.get('Return [%]'),
        'sharpe_ratio': sanitized_stats.get('Sharpe Ratio'),
        'max_drawdown_pct': sanitized_stats.get('Max. Drawdown [%]'),
        'win_rate_pct': sanitized_stats.get('Win Rate [%]'),
        'num_trades': sanitized_stats.get('# Trades'),
        'duration': str(sanitized_stats.get('Duration')),
    }

    results_file = os.path.join(results_dir, 'temp_result.json')
    with open(results_file, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print(f"Results saved to {results_file}")

    # Generate plot
    plot_file = os.path.join(results_dir, f'{strategy.__name__}.html')
    bt.plot(filename=plot_file)
    print(f"Plot saved to {plot_file}")

if __name__ == '__main__':
    run_backtest()
