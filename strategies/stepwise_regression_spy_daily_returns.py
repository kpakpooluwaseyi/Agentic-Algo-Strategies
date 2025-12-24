import pandas as pd
import numpy as np
import json
import os
from backtesting import Backtest, Strategy

# Helper function to sanitize stats for JSON serialization
def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object by converting non-serializable
    types to native Python types.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (np.int64, np.integer)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.float64, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        elif key == '_strategy' or key == '_equity_curve' or key == '_trades':
            continue  # Skip internal objects
        else:
            sanitized[key] = value
    return sanitized

def preprocess_data(filepath):
    """
    Loads 15m BTC data, cleans headers, and resamples it to a daily timeframe.
    """
    df = pd.read_csv(filepath, skipinitialspace=True)

    # Clean column headers: strip whitespace, remove trailing commas, and capitalize
    df.columns = [c.strip().rstrip(',').title() for c in df.columns]

    # Handle potential unnamed column from trailing comma
    if 'Unnamed: 6' in df.columns:
        df.drop(columns=['Unnamed: 6'], inplace=True)

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    # Resample to daily timeframe
    daily_df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Drop rows with NaN values that can occur from resampling
    daily_df.dropna(inplace=True)

    return daily_df

def ret(series, n):
    """
    Calculates the n-period return of a series.
    """
    return pd.Series(series).pct_change(n)

class StepwiseRegressionSpyDailyReturns(Strategy):
    """
    A mean-reversion strategy based on the finding that the 2-day
    return is a significant negative predictor of the next day's return.
    """

    def init(self):
        # Calculate the 2-day return as an indicator
        self.ret2 = self.I(ret, self.data.Close, 2)

    def next(self):
        # Enforce a 1-day holding period by closing any open position
        # at the start of the new bar.
        if self.position:
            self.position.close()

        # Get the most recent 2-day return. In `backtesting.py`, the `next`
        # method runs after the current bar's close, so `[-1]` gives us the
        # signal based on the most recently closed bar.
        previous_ret2 = self.ret2[-1]

        # Skip trading if the indicator is not yet mature
        if np.isnan(previous_ret2):
            return

        # Entry logic based on the mean-reversion principle
        if previous_ret2 > 0:
            # If the past 2 days were up, predict a down move (mean-reversion)
            self.sell()
        elif previous_ret2 < 0:
            # If the past 2 days were down, predict an up move (mean-reversion)
            self.buy()

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        # Preprocess the data to get daily bars
        daily_data = preprocess_data(data_path)

        # Instantiate the Backtest with increased initial cash
        bt = Backtest(daily_data, StepwiseRegressionSpyDailyReturns, cash=100000, commission=.002)

        # Run the backtest
        print("Running backtest...")
        stats = bt.run()
        print(stats)

        # Sanitize and save the results
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        sanitized = sanitize_stats(stats)

        with open(os.path.join(results_dir, 'temp_result.json'), 'w') as f:
            json.dump(sanitized, f, indent=4)

        print(f"\nResults saved to {results_dir}/temp_result.json")

        # Generate the plot
        plot_filename = os.path.join(results_dir, 'stepwise_regression_spy_daily_returns.html')
        try:
            bt.plot(filename=plot_filename)
            print(f"Plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot due to error: {e}")
