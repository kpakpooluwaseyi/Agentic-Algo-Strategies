import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import json
import os
import numpy as np

def sma_indicator(series: pd.Series, length: int):
    """
    Wrapper for pandas_ta.sma to be compatible with `self.I`.
    `self.I` passes a numpy array, so we convert it to a pandas Series.
    """
    return ta.sma(pd.Series(series), length=length)

class MovingAverageCrossReversal(Strategy):
    """
    A trend-following strategy that enters a long position when the price
    crosses above a moving average and a short position when it crosses below.
    It's a reversal system, meaning it's always in the market.
    """
    # Default parameters for the strategy
    ma_period = 50

    def init(self):
        """
        Initialize the strategy's indicators.
        """
        # Calculate the Simple Moving Average (SMA) using our wrapper
        self.sma = self.I(sma_indicator, self.data.Close, self.ma_period)

    def next(self):
        """
        Define the trading logic for each bar.
        """
        # Long entry condition: Price crosses above the SMA
        if crossover(self.data.Close, self.sma):
            # If we are currently short, close the position before going long.
            if self.position.is_short:
                self.position.close()
            # Enter a long position.
            self.buy()

        # Short entry condition: Price crosses below the SMA
        elif crossover(self.sma, self.data.Close):
            # If we are currently long, close the position before going short.
            if self.position.is_long:
                self.position.close()
            # Enter a short position.
            self.sell()

if __name__ == '__main__':
    # --- Data Loading ---
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure the file exists.")

    print(f"Loading data from: {data_path}")
    # Load data, ensuring datetime column is parsed correctly.
    data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    # Clean up column names: remove whitespace, handle trailing comma issue
    data.columns = data.columns.str.strip()
    # Drop unnamed columns that may be created by a trailing comma in the header
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Explicitly rename to the required format for backtesting.py
    rename_dict = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data = data.rename(columns=rename_dict)

    # --- Backtest Execution ---
    # Instantiate the backtest with the data, strategy, and initial conditions
    bt = Backtest(data, MovingAverageCrossReversal, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    print(stats)

    # --- Save Results ---
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats_obj):
        """A robust way to sanitize the stats object for JSON serialization."""
        sanitized = {}
        for key, value in stats_obj.items():
            # Skip non-serializable types like DataFrames or internal objects
            if isinstance(value, (pd.DataFrame, pd.Series)) or key.startswith('_'):
                continue
            # Convert numpy types to native Python types
            if isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value) if np.isfinite(value) else None
            elif pd.isna(value):
                sanitized[key] = None
            # Handle Timestamps and Timedeltas
            elif isinstance(value, (pd.Timestamp)):
                 sanitized[key] = value.isoformat()
            elif isinstance(value, (pd.Timedelta)):
                 sanitized[key] = str(value)
            else:
                sanitized[key] = value
        return sanitized

    final_stats = sanitize_stats(stats)
    final_stats['strategy_name'] = 'moving_average_cross_reversal'

    results_path = 'results/temp_result.json'
    with open(results_path, 'w') as f:
        json.dump(final_stats, f, indent=4)

    print(f"Backtest results saved to {results_path}")

    # --- Plotting ---
    plot_filename = 'results/moving_average_cross_reversal_plot.html'
    try:
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
