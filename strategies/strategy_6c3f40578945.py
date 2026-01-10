"""
Strategy: One Candle Setup (9:30 NY FVG Breakout)
Author: Moon Dev
Description: A time-specific breakout strategy that identifies the opening range of the
             New York session, waits for a breakout, and enters on a confirming
             Fair Value Gap (FVG). This implementation adapts the original 1m/5m
             forex/indices strategy for 15-minute BTC-USD data by defining a logical
             "NY session" start time.
"""
from backtesting import Strategy, Backtest
import pandas as pd
import numpy as np

# --- Helper Functions ---

def is_fvg(candle1, candle2, candle3, direction):
    """
    Identifies a Fair Value Gap (FVG) based on a three-candle sequence.

    Args:
        candle1, candle2, candle3 (pd.Series): Three consecutive candles
                                               (e.g., data.df.iloc[-3:]).
        direction (int): 1 for a bullish FVG, -1 for a bearish FVG.

    Returns:
        bool: True if an FVG is present, False otherwise.
    """
    if direction == 1:  # Bullish FVG
        return candle1['High'] < candle3['Low']
    elif direction == -1:  # Bearish FVG
        return candle1['Low'] > candle3['High']
    return False

# --- Strategy Class ---

# Note: The user's instructions specified inheriting from a non-existent `MoonDevStrategy`
# class. To ensure a runnable and functional backtest, this strategy inherits from the
# correct `backtesting.Strategy` class, following the repository's established patterns.
class OneCandleSetup(Strategy):
    """
    Implementation of the "One Candle Setup" strategy.

    State Machine Logic:
    1. WAITING_FOR_SESSION: Outside the defined trading session.
    2. RANGE_DEFINED: The first candle of the session has closed, defining the high/low range.
    3. BREAKOUT_CONFIRMED: A candle has closed outside the initial range, setting the trade direction.
    4. TRADE_EXECUTED: A position has been opened for the current session.
    """

    # --- Strategy Parameters ---
    # Define the logical "New York Open" session time in UTC
    session_start_hour = 13
    session_start_minute = 30

    # --- State Variables ---
    def init(self):
        # State tracking
        self.session_active = False
        self.range_high = None
        self.range_low = None
        self.breakout_direction = 0  # 1 for long, -1 for short, 0 for none
        self.trade_taken_this_session = False
        self.session_start_bar = -1

        # To handle time-based logic, pass time components as indicators
        self.hour = self.I(lambda: self.data.index.hour, name="hour")
        self.minute = self.I(lambda: self.data.index.minute, name="minute")

    def next(self):
        # Get current time from indicators
        current_bar_index = len(self.data) - 1
        current_hour = self.hour[-1]

        # --- Session Management ---
        # Look at the PREVIOUS candle's time to decide if a new session should START now.
        if len(self.data) < 2:
            return

        previous_hour = self.hour[-2]
        previous_minute = self.minute[-2]
        is_previous_candle_session_start = (previous_hour == self.session_start_hour and
                                            previous_minute == self.session_start_minute)

        # This block triggers at the close of the session's first candle (e.g., at 13:45 for a 13:30 candle)
        if is_previous_candle_session_start:
            self.session_active = True
            self.session_start_bar = len(self.data) - 2 # Index of the session's first bar
            self.trade_taken_this_session = False
            self.breakout_direction = 0

            # The range is defined by the candle that just closed, which is the session's first candle.
            self.range_high = self.data.High[-1]
            self.range_low = self.data.Low[-1]

        # Deactivate session logic a few hours after start to reset for the next day
        if self.session_active and current_hour > self.session_start_hour + 4:
            self.session_active = False
            self.range_high = None # Clear the range
            self.range_low = None

        # --- Session Timeout ---
        # Close any open trade after 90 minutes (6 candles * 15 min)
        if self.position and self.session_start_bar > 0:
            bars_since_session_start = current_bar_index - self.session_start_bar
            if bars_since_session_start > 6:
                self.position.close()

        # --- Trading Logic (only executes if a session is active and no trade has been taken) ---
        if not self.session_active or self.range_high is None or self.trade_taken_this_session or self.position:
            return

        # --- Step 1: Wait for Breakout ---
        if self.breakout_direction == 0:
            if self.data.Close[-1] > self.range_high:
                self.breakout_direction = 1  # Bullish breakout
            elif self.data.Close[-1] < self.range_low:
                self.breakout_direction = -1 # Bearish breakout
            return # Wait for the next candle to confirm FVG

        # --- Step 2: Confirm with FVG and Enter Trade ---
        # Ensure we have enough data for a 3-candle pattern
        if len(self.data.Close) < 3:
            return

        candle1 = self.data.df.iloc[-3]
        candle2 = self.data.df.iloc[-2]
        candle3 = self.data.df.iloc[-1] # The most recently closed candle

        # Bullish Entry Logic
        if self.breakout_direction == 1:
            # Check for a bullish FVG that is entirely above the range
            fvg_is_valid = is_fvg(candle1, candle2, candle3, direction=1)
            fvg_is_outside_range = candle1['Low'] > self.range_high

            if fvg_is_valid and fvg_is_outside_range:
                # --- Long Entry Execution ---
                entry_price = self.data.Close[-1]
                sl = candle2['Low']
                risk = entry_price - sl
                tp = entry_price + (risk * 2)

                # Ensure SL and TP are valid before placing trade
                if risk > 0 and tp > entry_price:
                    self.buy(sl=sl, tp=tp)
                    self.trade_taken_this_session = True

        # Bearish Entry Logic
        elif self.breakout_direction == -1:
            # Check for a bearish FVG that is entirely below the range
            fvg_is_valid = is_fvg(candle1, candle2, candle3, direction=-1)
            fvg_is_outside_range = candle1['High'] < self.range_low

            if fvg_is_valid and fvg_is_outside_range:
                # --- Short Entry Execution ---
                entry_price = self.data.Close[-1]
                sl = candle2['High']
                risk = sl - entry_price
                tp = entry_price - (risk * 2)

                # Ensure SL and TP are valid before placing trade
                if risk > 0 and tp < entry_price:
                    self.sell(sl=sl, tp=tp)
                    self.trade_taken_this_session = True

# --- Main execution block (for standalone testing) ---
if __name__ == '__main__':
    import json
    import os

    def sanitize_stats(stats):
        """
        Sanitizes the backtest stats object to be JSON serializable,
        handling specific pandas and numpy types.
        """
        if stats is None:
            return None

        # Convert pandas Series to a standard dictionary
        clean_stats = stats.to_dict()

        # Remove non-serializable objects
        if '_strategy' in clean_stats:
            del clean_stats['_strategy']
        if '_equity_curve' in clean_stats:
            del clean_stats['_equity_curve']
        if '_trades' in clean_stats:
            del clean_stats['_trades']

        # Convert specific types to JSON-friendly formats
        for key, value in clean_stats.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                clean_stats[key] = str(value)
            elif isinstance(value, (np.integer, np.int64)):
                clean_stats[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                clean_stats[key] = float(value)
            elif pd.isna(value):
                clean_stats[key] = None

        return clean_stats

    # --- Data Loading and Preparation ---
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        # Sanitize column names (e.g., remove leading/trailing spaces and capitalize)
        data.columns = [col.strip().capitalize() for col in data.columns]
    except FileNotFoundError:
        print("Error: 'data/BTC-USD-15m.csv' not found.")
        print("Please ensure the data file is in the correct directory.")
        exit()

    # --- Backtest Execution ---
    bt = Backtest(
        data,
        OneCandleSetup,
        cash=100_000,
        commission=.002
    )

    stats = bt.run()

    # --- Results Output ---
    print("\n--- Backtest Results ---")
    print(stats)

    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save plot
    plot_filename = 'results/strategy_6c3f40578945_plot.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"\nPlot saved to {plot_filename}")

    # Save stats to JSON
    stats_filename = 'results/temp_result.json'
    sanitized_results = sanitize_stats(stats)
    with open(stats_filename, 'w') as f:
        json.dump(sanitized_results, f, indent=4)
    print(f"Stats saved to {stats_filename}")
