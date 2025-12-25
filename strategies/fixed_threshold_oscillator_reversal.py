import talib
import pandas as pd
from backtesting import Backtest, Strategy
import json
import os

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it's JSON serializable.
    Removes non-serializable objects like _strategy, _equity_curve, _trades.
    """
    if stats is None:
        return {}

    # Convert stats object to a dictionary if it's not already
    stats_dict = dict(stats) if not isinstance(stats, dict) else stats

    # List of keys to remove
    keys_to_remove = ['_strategy', '_equity_curve', '_trades']
    for key in keys_to_remove:
        stats_dict.pop(key, None)

    # Sanitize remaining values
    for key, value in list(stats_dict.items()):
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)
        elif pd.isna(value):
            stats_dict[key] = None
        # Add any other type conversions if necessary

    return stats_dict

class FixedThresholdOscillatorReversal(Strategy):
    """
    A mean-reversion strategy that enters a long position when an oscillator
    crosses above a fixed threshold and a short position when it crosses below.
    It's a continuous reversal rule, meaning the strategy is always in the market.
    """
    rsi_period = 14
    threshold = 75

    def init(self):
        # Initialize the RSI indicator
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)

    def next(self):
        # If RSI is above the threshold
        if self.rsi[-1] > self.threshold:
            # If we are short, close the position
            if self.position.is_short:
                self.position.close()
            # If we are not already long, go long
            if not self.position.is_long:
                self.buy()

        # If RSI is at or below the threshold
        elif self.rsi[-1] <= self.threshold:
            # If we are long, close the position
            if self.position.is_long:
                self.position.close()
            # If we are not already short, go short
            if not self.position.is_short:
                self.sell()

if __name__ == '__main__':
    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    strategy_name = 'fixed_threshold_oscillator_reversal'
    output_dir = 'results'

    # --- Data Loading ---
    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Clean and standardize column names
        data.columns = [c.strip().title() for c in data.columns]
        # Drop any unnamed columns that may have been created by a trailing comma in the CSV header
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        print(f"Data loaded successfully from {data_path}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # As a fallback for testing, create some synthetic data
        print("Generating synthetic data for demonstration...")
        from backtesting.test import GOOG
        data = GOOG.iloc[-2000:] # Use a slice of sample data

    # --- Backtesting ---
    # Use FractionalBacktest to allow for fractional position sizes
    from backtesting.lib import FractionalBacktest
    bt = FractionalBacktest(data, FixedThresholdOscillatorReversal, cash=10_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    print("\n--- Backtest Results ---")
    print(stats)

    # --- Save Results ---
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    plot_filename = os.path.join(output_dir, f'{strategy_name}.html')
    try:
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not save plot: {e}")

    # Save stats to JSON
    json_filename = os.path.join(output_dir, 'temp_result.json')
    try:
        # Sanitize stats for JSON serialization
        sanitized_results = sanitize_stats(stats)
        with open(json_filename, 'w') as f:
            json.dump(sanitized_results, f, indent=4)
        print(f"Stats saved to {json_filename}")
    except Exception as e:
        print(f"Could not save stats to JSON: {e}")
