from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import json
import os

# Helper function to convert backtesting.py's numpy arrays to pandas Series for pandas_ta compatibility
def array_to_series(func, *args, **kwargs):
    """
    Wrapper to convert numpy array inputs from backtesting.py into pandas Series
    for compatibility with libraries like pandas_ta.
    """
    series_args = [pd.Series(arg) for arg in args]
    # The result from pandas_ta can be a DataFrame (e.g., macd) or a Series (e.g., mfi)
    result = func(*series_args, **kwargs)
    if isinstance(result, pd.DataFrame):
        # For DataFrames, return a tuple of numpy arrays for each column
        return tuple(res.values for _, res in result.items())
    else:
        # For Series, return the numpy array of its values
        return result.values

class Strategy2f80e364e70d(Strategy):
    """
    Strategy based on a simplified interpretation of Market Cipher B concepts,
    using MACD for momentum and MFI for money flow.
    - Long Entry: Bullish MACD crossover when MFI is oversold.
    - Short Entry: Bearish MACD crossover when MFI is overbought.
    """
    # --- Strategy Parameters ---
    # MACD settings
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # MFI settings
    mfi_period = 14
    mfi_oversold = 30
    mfi_overbought = 70

    # Risk management
    sl_pct = 2.0  # Stop loss percentage
    tp_pct = 3.0  # Take profit percentage (for a 1.5 R:R)

    def init(self):
        """
        Initialize the indicators for the strategy.
        Since src.indicators.vumanchu is not available, we use pandas_ta as a proxy
        for MACD and MFI, which are common components of such strategies.
        """
        # MACD Indicator
        self.macd, self.macd_hist, self.macd_signal = self.I(
            array_to_series,
            ta.macd,
            self.data.Close,
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
            name="MACD"
        )

        # Money Flow Index (MFI) Indicator
        self.mfi = self.I(
            array_to_series,
            ta.mfi,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume,
            length=self.mfi_period,
            name="MFI"
        )

    def next(self):
        """
        Define the entry and exit logic for the strategy on each bar.
        """
        # --- Risk Management: Do not enter new trades if a position is already open ---
        if self.position:
            return

        # --- Entry Conditions ---
        price = self.data.Close[-1]
        sl = self.sl_pct / 100
        tp = self.tp_pct / 100

        # Long Entry Signal: Bullish MACD crossover + MFI oversold
        is_long_signal = (crossover(self.macd, self.macd_signal) and
                          self.mfi[-1] < self.mfi_oversold)

        # Short Entry Signal: Bearish MACD crossover + MFI overbought
        is_short_signal = (crossover(self.macd_signal, self.macd) and
                           self.mfi[-1] > self.mfi_overbought)

        # --- Trade Execution ---
        if is_long_signal:
            # Calculate stop loss and take profit levels
            stop_loss = price * (1 - sl)
            take_profit = price * (1 + tp)
            self.buy(sl=stop_loss, tp=take_profit)

        elif is_short_signal:
            # Calculate stop loss and take profit levels
            stop_loss = price * (1 + sl)
            take_profit = price * (1 - tp)
            self.sell(sl=stop_loss, tp=take_profit)


def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to make it JSON serializable,
    handling various data types that may cause issues.
    """
    if stats is None:
        return {}

    # Convert pandas Series to dictionary
    sanitized = stats.to_dict()

    # List of keys to remove that are not JSON serializable or useful
    keys_to_remove = ['_strategy', '_equity_curve', '_trades']
    for key in keys_to_remove:
        sanitized.pop(key, None)

    # Convert specific types to JSON-friendly formats
    for key, value in sanitized.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif pd.isna(value):
            sanitized[key] = None
        elif isinstance(value, (int, float, str, bool)) or value is None:
            continue # Already JSON serializable
        else:
            # Convert numpy types to native Python types
            try:
                sanitized[key] = value.item()
            except AttributeError:
                sanitized[key] = str(value) # Fallback for other complex types

    return sanitized


if __name__ == '__main__':
    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    strategy_name = 'strategy_2f80e364e70d'
    output_dir = 'results'

    # --- Data Loading ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}. Please ensure the data is in the correct location.")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Sanitize column names to match backtesting.py's expected format (Titlecase)
    data.columns = [c.strip().title() for c in data.columns]

    # --- Backtesting ---
    print(f"Running backtest for {strategy_name}...")
    bt = Backtest(data, Strategy2f80e364e70d, cash=100_000, commission=.002)
    stats = bt.run()

    # --- Results ---
    print("\n--- Backtest Results ---")
    print(stats)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save sanitized stats to JSON
    json_path = os.path.join(output_dir, 'temp_result.json')
    sanitized = sanitize_stats(stats)
    with open(json_path, 'w') as f:
        json.dump(sanitized, f, indent=4)
    print(f"\nSaved statistics to {json_path}")

    # Generate and save plot
    plot_path = os.path.join(output_dir, f'{strategy_name}_plot.html')
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
