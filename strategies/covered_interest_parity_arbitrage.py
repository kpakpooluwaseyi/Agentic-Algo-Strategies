# STRATEGY: covered_interest_parity_arbitrage
# DESCRIPTION: This strategy is a proxy for Covered Interest Parity Arbitrage.
# A direct implementation is not possible due to the limitations of the backtesting environment,
# which only provides BTC-USD data and does not include the necessary data for a true CIP arbitrage strategy
# (i.e., forward exchange rates and interest rates for two different currencies).
# This proxy strategy attempts to capture the spirit of arbitrage by identifying and exploiting
# what could be considered "mispricings" in the market. It does this by implementing a mean-reversion strategy
# using Bollinger Bands. The strategy assumes that when the price deviates significantly from its moving average
# (i.e., touches the upper or lower Bollinger Band), it is temporarily "mispriced" and will likely revert to the mean.

import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def PtaBollingerBands(close_series, length=20, std=2.0):
    """
    Wrapper for pandas_ta.bbands that returns a tuple of numpy arrays
    as expected by backtesting.py's self.I().
    """
    bbands = ta.bbands(close=pd.Series(close_series), length=length, std=std)
    # The library in this environment appears to append the std dev value twice.
    lower_col = f'BBL_{length}_{std}'
    middle_col = f'BBM_{length}_{std}'
    upper_col = f'BBU_{length}_{std}'

    # In newer versions of pandas_ta, the column names might not have the duplicate std dev
    if lower_col not in bbands:
        lower_col = f'BBL_{length}_{std}.0'
        middle_col = f'BBM_{length}_{std}.0'
        upper_col = f'BBU_{length}_{std}.0'

    # Handle cases where column names might differ
    if not all(col in bbands.columns for col in [lower_col, middle_col, upper_col]):
        # Fallback to dynamic column finding if specific names fail
        try:
            lower_col = [col for col in bbands.columns if col.startswith('BBL_')][0]
            middle_col = [col for col in bbands.columns if col.startswith('BBM_')][0]
            upper_col = [col for col in bbands.columns if col.startswith('BBU_')][0]
        except IndexError:
            raise KeyError("Could not find Bollinger Bands columns in the DataFrame.")

    return bbands[lower_col].values, bbands[middle_col].values, bbands[upper_col].values

def generate_synthetic_data():
    """Generates synthetic data for testing the strategy."""
    print("Generating synthetic data...")
    n_points = 2000
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))
    price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.1)
    price += np.sin(np.linspace(0, 200, n_points)) * 2
    data = pd.DataFrame({
        'Open': price, 'High': price * 1.005, 'Low': price * 0.995,
        'Close': price, 'Volume': np.random.randint(100, 1000, n_points)
    }, index=index)
    return data

class CoveredInterestParityArbitrage(Strategy):
    """
    A mean-reversion strategy that uses Bollinger Bands.
    - Enters long when the price crosses below the lower band.
    - Enters short when the price crosses above the upper band.
    - Exits when the price reverts to the middle band.
    """
    bb_period = 20
    bb_std_dev = 2.0

    def init(self):
        self.lower_band, self.middle_band, self.upper_band = self.I(
            PtaBollingerBands, self.data.Close, self.bb_period, self.bb_std_dev
        )

    def next(self):
        if self.position:
            if self.position.is_long and self.data.Close[-1] >= self.middle_band[-1]:
                self.position.close()
            elif self.position.is_short and self.data.Close[-1] <= self.middle_band[-1]:
                self.position.close()
        else:
            if crossover(self.lower_band, self.data.Close):
                self.buy()
            elif crossover(self.data.Close, self.upper_band):
                self.sell()

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        try:
            # Load data, ensuring correct headers and parsing dates.
            data = pd.read_csv(
                data_path,
                index_col='datetime',
                parse_dates=True,
                header=0,
                names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unnamed'],
                usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'.")
        data = generate_synthetic_data()

    bt = Backtest(data, CoveredInterestParityArbitrage, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)): continue
            if isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, int): sanitized[key] = int(value)
            elif isinstance(value, pd.Timestamp): sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta): sanitized[key] = str(value)
            elif pd.isna(value): sanitized[key] = None
            elif key.startswith('_'): continue
            else: sanitized[key] = value
        return sanitized

    final_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    try:
        plot_filename = 'results/covered_interest_parity_arbitrage.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
