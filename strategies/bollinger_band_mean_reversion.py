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
    lower_col = f'BBL_{length}_{std}_{std}'
    middle_col = f'BBM_{length}_{std}_{std}'
    upper_col = f'BBU_{length}_{std}_{std}'
    return bbands[lower_col], bbands[middle_col], bbands[upper_col]

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

class BollingerBandMeanReversion(Strategy):
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
                self.buy(size=1)
            elif crossover(self.data.Close, self.upper_band):
                self.sell(size=1)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        try:
            data = pd.read_csv(
                data_path, index_col='datetime', parse_dates=True, header=0,
                names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unnamed'],
                usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'.")
        data = generate_synthetic_data()

    bt = Backtest(data, BollingerBandMeanReversion, cash=100_000, commission=.002)

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
        plot_filename = 'results/bollinger_band_mean_reversion.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
