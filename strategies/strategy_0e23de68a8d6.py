from backtesting import Backtest, Strategy
import pandas as pd
import pandas_ta as ta
import json
import os
import numpy as np

# --- Strategy Definition ---

class VpaStrategy(Strategy):
    """
    Volume Price Analysis (VPA) strategy to detect market reversals
    based on volume climaxes. This strategy identifies bars with exceptionally
    high volume (potential climaxes) and waits for a subsequent confirmation
    bar that reverses the direction, signaling that institutional players have
    absorbed the pressure.
    """
    # --- Strategy Parameters ---
    volume_lookback = 20      # Lookback period for the volume moving average
    volume_threshold_pct = 200  # Pct above VMA to be considered a climax
    confirmation_bars = 3     # How many bars to wait for a reversal
    sl_buffer_pct = 0.02      # Stop loss buffer (2%)
    tp_rr = 2.0               # Take profit risk-reward ratio

    def init(self):
        """
        Initialize the strategy's indicators and state variables.
        """
        # Calculate Volume Moving Average using pandas_ta
        self.volume_ma = self.I(
            ta.sma,
            pd.Series(self.data.Volume),
            length=self.volume_lookback
        )

        # State variables to track climaxes
        self.climax_bar_index = None
        self.climax_type = None  # 'up' for selling climax, 'down' for buying climax
        self.climax_high = None
        self.climax_low = None

    def next(self):
        """
        The main strategy logic that runs on each bar.
        """
        # If a position is already open, do not execute new logic
        if self.position:
            return

        current_index = len(self.data.Close) - 1

        # --- 1. Check for reversal confirmation if a climax is active ---
        if self.climax_bar_index is not None:
            bars_since_climax = current_index - self.climax_bar_index

            # A. Handle Selling Climax (high-volume UP bar) confirmation
            if self.climax_type == 'up':
                # Check for bearish reversal confirmation
                if self.data.Close[-1] < self.climax_low:
                    sl = self.climax_high * (1 + self.sl_buffer_pct / 100)
                    risk = sl - self.data.Close[-1]
                    tp = self.data.Close[-1] - risk * self.tp_rr
                    if tp > 0: # Ensure take profit is valid
                        self.sell(sl=sl, tp=tp)
                    self.climax_bar_index = None # Reset state

            # B. Handle Buying Climax (high-volume DOWN bar) confirmation
            elif self.climax_type == 'down':
                # Check for bullish reversal confirmation
                if self.data.Close[-1] > self.climax_high:
                    sl = self.climax_low * (1 - self.sl_buffer_pct / 100)
                    risk = self.data.Close[-1] - sl
                    tp = self.data.Close[-1] + risk * self.tp_rr
                    self.buy(sl=sl, tp=tp)
                    self.climax_bar_index = None # Reset state

            # C. Invalidate climax if no confirmation within the window
            if self.climax_bar_index and bars_since_climax >= self.confirmation_bars:
                self.climax_bar_index = None

        # --- 2. Check for a new climax event if none is active ---
        if self.climax_bar_index is None:
            volume_threshold = self.volume_ma[-1] * (1 + self.volume_threshold_pct / 100)

            # Check if current volume exceeds the threshold
            if self.data.Volume[-1] > volume_threshold:
                # A. Selling Climax: A high-volume UP bar
                if self.data.Close[-1] > self.data.Open[-1]:
                    self.climax_type = 'up'
                    self.climax_bar_index = current_index
                    self.climax_high = self.data.High[-1]
                    self.climax_low = self.data.Low[-1]

                # B. Buying Climax: A high-volume DOWN bar
                elif self.data.Close[-1] < self.data.Open[-1]:
                    self.climax_type = 'down'
                    self.climax_bar_index = current_index
                    self.climax_high = self.data.High[-1]
                    self.climax_low = self.data.Low[-1]

# --- Main Execution Block ---

if __name__ == '__main__':
    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    output_dir = 'results'
    results_file = os.path.join(output_dir, 'temp_result.json')
    plot_file = os.path.join(output_dir, 'strategy_0e23de68a8d6.html')

    # --- Data Loading ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # The CSV header has a trailing comma, creating an unnamed column.
    # We also need to standardize column names to 'Capitalized'.
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # --- Backtesting ---
    bt = Backtest(data, VpaStrategy, cash=100_000, commission=.002)
    stats = bt.run()

    # --- Output Results ---
    os.makedirs(output_dir, exist_ok=True)

    def sanitize_stats(stats):
        """Prepares the backtesting stats for JSON serialization."""
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame, Strategy)):
                continue
            if pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (int, float, str, bool)):
                 sanitized[key] = value
            elif isinstance(value, (np.int64, np.int32)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                sanitized[key] = float(value)
            else:
                sanitized[key] = str(value)
        return sanitized

    clean_stats = sanitize_stats(stats)

    # Save statistics to a JSON file
    with open(results_file, 'w') as f:
        json.dump(clean_stats, f, indent=4)

    print(f"Backtest statistics saved to {results_file}")
    print(stats)

    # Generate and save the plot
    try:
        bt.plot(filename=plot_file, open_browser=False)
        print(f"Backtest plot saved to {plot_file}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
