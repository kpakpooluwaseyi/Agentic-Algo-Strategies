import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta as ta
import json
import os

# --- VUMANCHU INDICATOR PROXY ---
# The user's request specified using a custom indicator from `src.indicators.vumanchu` to represent
# the "Market Cipher B blue momentum wave". However, this file does not exist in the repository.
# Following the established development pattern for this codebase, the standard procedure is to
# substitute a functionally similar indicator from a standard library.
#
# For this strategy, the Money Flow Index (MFI) from the `pandas_ta` library has been chosen as a proxy.
# MFI is a momentum oscillator that incorporates both price and volume, making it a suitable
# and robust replacement for the described behavior of the Market Cipher B momentum wave.
# ---

def MFI_indicator(series, period):
    """
    Custom indicator function to calculate Money Flow Index (MFI).
    This function wraps the pandas_ta MFI implementation for use with backtesting.py's self.I().
    """
    return series.ta.mfi(length=period).values


class MarketCipherBOverboughtContinuationLong(Strategy):
    """
    This strategy enters a long position when a momentum indicator (MFI) signals an extreme
    overbought condition, anticipating further trend continuation. It exits when the
    momentum subsides. This approach is based on the concept of riding "acceleration phases"
    in strong uptrends, as described for Market Cipher B on higher timeframes.
    """
    mfi_period = 14
    overbought_threshold = 60
    stop_loss_pct = 0.05  # 5% stop loss

    def init(self):
        # Calculate the Money Flow Index (MFI) as a proxy for the Market Cipher B momentum wave.
        self.mfi = self.I(MFI_indicator, self.data.df, self.mfi_period)

    def next(self):
        # Entry condition:
        # If not already in a position, check if the MFI has just crossed above the overbought threshold.
        if not self.position:
            if crossover(self.mfi, self.overbought_threshold):
                # Calculate stop loss price
                sl_price = self.data.Close[-1] * (1 - self.stop_loss_pct)
                self.buy(sl=sl_price)

        # Exit condition:
        # If in a position, check if the MFI has crossed back below the overbought threshold.
        elif self.mfi[-1] < self.overbought_threshold:
            self.position.close()


def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it is JSON serializable.
    Removes non-serializable types like DataFrame, Timestamps, and Timedeltas.
    """
    if stats is None:
        return {}

    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            # Skip DataFrame/Series objects like _equity_curve and _trades
            continue
        elif isinstance(value, (int, float, str, bool)) or value is None:
            sanitized[key] = value
        else:
            try:
                # Attempt to convert other types to a serializable format
                json.dumps({key: value})
                sanitized[key] = value
            except (TypeError, OverflowError):
                sanitized[key] = str(value) # Fallback to string representation
    return sanitized

if __name__ == '__main__':
    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    results_filename = 'temp_result.json'
    plot_filename = 'results/strategy_88ed0a33c3bb.html'

    # --- Data Loading ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}. Please ensure the data is available.")

    data = pd.read_csv(
        data_path,
        parse_dates=['datetime'],
        index_col='datetime'
    )

    # Sanitize column names for backtesting.py
    data.columns = [col.strip().capitalize() for col in data.columns]
    data.rename(columns={'Volume': 'Volume'}, inplace=True) # Ensure volume is also capitalized if it exists

    # --- Backtesting ---
    bt = Backtest(data, MarketCipherBOverboughtContinuationLong, cash=100000, commission=.002)

    print("Running backtest...")
    stats = bt.run()

    # --- Results ---
    print("\nBacktest Results:")
    print(stats)

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Save sanitized stats to JSON
    sanitized_stats = sanitize_stats(stats)
    results_path = os.path.join(results_dir, results_filename)
    with open(results_path, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print(f"\nSaved strategy results to {results_path}")

    # Generate and save plot
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Saved plot to {plot_filename}")
    except Exception as e:
        print(f"\nCould not generate plot due to an error: {e}")
        print("This may be due to plotting library issues. The statistical results are saved.")
