
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import json
import os

def rsi(series: pd.Series, n: int) -> pd.Series:
    """Computes the Relative Strength Index (RSI) using pandas_ta."""
    return ta.rsi(series, length=n)

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it's JSON-serializable.
    It converts numpy and pandas types to native Python types and removes
    non-serializable objects.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(key, str) and key.startswith('_'):
            continue  # Skip private attributes

        if isinstance(value, (pd.DataFrame, pd.Series, Strategy)):
            sanitized[key] = None  # Remove non-serializable objects
        elif pd.isna(value):
            sanitized[key] = None  # Convert NaN to None
        elif isinstance(value, (int, float, bool, str, type(None))):
            sanitized[key] = value  # Keep basic types
        elif hasattr(value, 'item'):
            sanitized[key] = value.item()  # Convert numpy types to Python native types
        else:
            sanitized[key] = str(value) # Convert other types to string as a fallback
    return sanitized

class ContrarianCyclePsychology(Strategy):
    """
    Implements a contrarian strategy based on market psychology extremes,
    using RSI as a proxy for sentiment.

    Long Entry: Enters a long position when the market is perceived as "fearful"
                (RSI is oversold).
    Short Entry: Enters a short position when the market is perceived as "greedy"
                 (RSI is overbought).
    """
    # Optimizable parameters
    rsi_period = 14
    oversold_threshold = 30
    overbought_threshold = 70
    rr_ratio = 1.5
    sl_buffer_pct = 0.01  # 1% buffer for stop-loss

    def init(self):
        """
        Initialize indicators.
        """
        # Calculate RSI on the close prices
        self.rsi = self.I(rsi, pd.Series(self.data.Close), self.rsi_period)

    def next(self):
        """
        Define the trading logic for each bar.
        """
        # Skip if we don't have enough data for the RSI calculation
        if len(self.data.Close) < self.rsi_period:
            return

        # --- LONG ENTRY ---
        # If not in a position and RSI crosses below the oversold threshold
        if not self.position and self.rsi[-1] < self.oversold_threshold:
            # Calculate stop-loss with a buffer below the low of the entry candle
            stop_loss = self.data.Low[-1] * (1 - self.sl_buffer_pct)
            entry_price = self.data.Close[-1]

            # Calculate take-profit based on the risk-reward ratio
            risk = entry_price - stop_loss
            take_profit = entry_price + risk * self.rr_ratio

            # Place the buy order if risk is positive
            if risk > 0:
                self.buy(sl=stop_loss, tp=take_profit)

        # --- SHORT ENTRY ---
        # If not in a position and RSI crosses above the overbought threshold
        elif not self.position and self.rsi[-1] > self.overbought_threshold:
            # Calculate stop-loss with a buffer above the high of the entry candle
            stop_loss = self.data.High[-1] * (1 + self.sl_buffer_pct)
            entry_price = self.data.Close[-1]

            # Calculate take-profit based on the risk-reward ratio
            risk = stop_loss - entry_price
            take_profit = entry_price - risk * self.rr_ratio

            # Place the sell order if risk is positive
            if risk > 0:
                self.sell(sl=stop_loss, tp=take_profit)


if __name__ == '__main__':
    # Define the path to the data
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        # Load the data
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

        # Clean data: drop unnamed columns, strip whitespace from column names, and then capitalize.
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        data.columns = [c.strip().capitalize() for c in data.columns]

        # Initialize and run the backtest
        bt = Backtest(data, ContrarianCyclePsychology, cash=100000, commission=.002)

        print("Running backtest with default parameters...")
        stats = bt.run()
        print(stats)

        # Ensure the results directory exists
        os.makedirs('results', exist_ok=True)

        # Save numerical results to JSON
        stats_dict = sanitize_stats(stats.to_dict())
        with open('results/temp_result.json', 'w') as f:
            json.dump(stats_dict, f, indent=4)
        print("\nBacktest stats saved to results/temp_result.json")

        # Generate and save the plot
        plot_filename = 'results/contrarian_cycle_psychology.html'
        try:
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Backtest plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
