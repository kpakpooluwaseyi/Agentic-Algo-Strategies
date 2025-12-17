
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply
import numpy as np
import json
import os

def sanitize_stats(stats):
    """
    Sanitizes the stats object by converting non-serializable types to strings or basic types.
    Removes the _strategy object to avoid serialization issues.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(key, str) and key.startswith('_'):
            continue
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.int64, np.float64)):
            sanitized[key] = value.item()
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            continue
        else:
            sanitized[key] = value
    return sanitized

def preprocess_data(filepath='data/BTC-USD-15m.csv'):
    """
    Loads 15-minute BTC data and adds the previous day's high and low.
    """
    if not os.path.exists(filepath):
        print(f"Data file not found at {filepath}. Generating synthetic data.")
        # Generate synthetic data if the file doesn't exist
        date_rng = pd.date_range(start='2023-01-01', end='2023-03-01', freq='15min')
        price = 20000 + (np.random.randn(len(date_rng)).cumsum() * 10)
        volume = np.random.randint(100, 1000, size=len(date_rng))
        df = pd.DataFrame(date_rng, columns=['datetime'])
        df['open'] = price
        df['high'] = price + np.random.uniform(0, 10, size=len(date_rng))
        df['low'] = price - np.random.uniform(0, 10, size=len(date_rng))
        df['close'] = price + np.random.uniform(-5, 5, size=len(date_rng))
        df['volume'] = volume
    else:
        df = pd.read_csv(filepath)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Calculate daily high and low
    daily_resample = df['High'].resample('D')
    daily_high = daily_resample.max()
    daily_low = df['Low'].resample('D').min()

    # Shift to get the *previous* day's data to avoid lookahead bias
    df['prev_daily_high'] = daily_high.shift(1).reindex(df.index, method='ffill')
    df['prev_daily_low'] = daily_low.shift(1).reindex(df.index, method='ffill')

    # Drop only rows where the new columns are NaN (i.e., the first day of data)
    df.dropna(subset=['prev_daily_high', 'prev_daily_low'], inplace=True)
    return df

# --- Strategy Definition ---
class PredictableCandleExpansion(Strategy):
    """
    This strategy is based on Smart Money Concepts (SMC). It aims to identify a
    higher timeframe directional bias by observing how price reacts to key levels,
    specifically the previous day's high and low.

    1.  **Liquidity Sweep:** It waits for the price to run the liquidity resting
        above the previous day's high or below the previous day's low.
    2.  **Displacement & FVG:** After the sweep, it looks for a strong, fast
        move in the opposite direction that creates a Fair Value Gap (FVG).
        An FVG is a 3-candle pattern where there is an inefficiency or imbalance.
    3.  **Entry:** It enters a trade when the price retraces back into this
        newly formed FVG, anticipating the continuation of the expansion move.
    """
    # --- Strategy Parameters ---
    risk_reward_ratio = 3.0

    # --- State Management ---
    def init(self):
        # State machine: 0=Searching, 1=Displacement Lookout, 2=Entry Waiting
        self.trade_state = 0
        self.setup_direction = 0  # -1 for short, 1 for long

        # Setup variables
        self.sweep_level = None
        self.fvg_high = None
        self.fvg_low = None
        self.stop_loss = None

        # Invalidation counter to prevent stale setups
        self.invalidation_counter = 0
        self.max_wait_bars = 20 # Max bars to wait for displacement or entry

    def next(self):
        # Ensure we don't place new trades if a position is already open
        if self.position:
            return

        # --- State Machine Logic ---

        # State 0: Searching for a liquidity sweep of previous day's H/L
        if self.trade_state == 0:
            # Check for sweep of previous day's high (potential short setup)
            if self.data.High[-1] > self.data.prev_daily_high[-1]:
                self.trade_state = 1
                self.setup_direction = -1
                self.sweep_level = self.data.High[-1]
                self.invalidation_counter = 0

            # Check for sweep of previous day's low (potential long setup)
            elif self.data.Low[-1] < self.data.prev_daily_low[-1]:
                self.trade_state = 1
                self.setup_direction = 1
                self.sweep_level = self.data.Low[-1]
                self.invalidation_counter = 0

        # State 1: A sweep occurred. Looking for displacement and a Fair Value Gap (FVG)
        elif self.trade_state == 1:
            self.invalidation_counter += 1
            if self.invalidation_counter > self.max_wait_bars:
                self.reset_state()
                return

            # --- Correct FVG Detection ---
            # A 3-candle pattern where candle 2 (-2) leaves an imbalance.
            # We check on the close of candle 3 (-1).
            first_candle_low = self.data.Low[-3]
            first_candle_high = self.data.High[-3]
            third_candle_high = self.data.High[-1]
            third_candle_low = self.data.Low[-1]

            # For a SHORT setup, we need a BEARISH FVG
            # The high of the 3rd candle is below the low of the 1st candle.
            if self.setup_direction == -1:
                is_bearish_fvg = third_candle_high < first_candle_low
                if is_bearish_fvg:
                    self.fvg_high = first_candle_low    # Top of FVG
                    self.fvg_low = third_candle_high   # Bottom of FVG
                    self.stop_loss = self.sweep_level
                    self.trade_state = 2
                    self.invalidation_counter = 0

            # For a LONG setup, we need a BULLISH FVG
            # The low of the 3rd candle is above the high of the 1st candle.
            elif self.setup_direction == 1:
                is_bullish_fvg = third_candle_low > first_candle_high
                if is_bullish_fvg:
                    self.fvg_high = third_candle_low     # Top of FVG
                    self.fvg_low = first_candle_high    # Bottom of FVG
                    self.stop_loss = self.sweep_level
                    self.trade_state = 2
                    self.invalidation_counter = 0

        # State 2: FVG formed. Waiting for price to retrace for a limit order entry
        elif self.trade_state == 2:
            self.invalidation_counter += 1
            if self.invalidation_counter > self.max_wait_bars:
                self.reset_state()
                return

            # Check for SHORT entry: price wicks up into FVG
            if self.setup_direction == -1:
                if self.data.High[-1] > self.fvg_low:
                    entry_price = self.fvg_low # Place limit order at bottom of FVG
                    if self.stop_loss > entry_price:
                        risk = self.stop_loss - entry_price
                        take_profit = entry_price - (risk * self.risk_reward_ratio)
                        self.sell(limit=entry_price, sl=self.stop_loss, tp=take_profit)
                    self.reset_state() # Reset after attempt

            # Check for LONG entry: price wicks down into FVG
            elif self.setup_direction == 1:
                if self.data.Low[-1] < self.fvg_high:
                    entry_price = self.fvg_high # Place limit order at top of FVG
                    if self.stop_loss < entry_price:
                        risk = entry_price - self.stop_loss
                        take_profit = entry_price + (risk * self.risk_reward_ratio)
                        self.buy(limit=entry_price, sl=self.stop_loss, tp=take_profit)
                    self.reset_state() # Reset after attempt

    def reset_state(self):
        """Resets the state machine and all setup variables."""
        self.trade_state = 0
        self.setup_direction = 0
        self.sweep_level = None
        self.fvg_high = None
        self.fvg_low = None
        self.stop_loss = None
        self.invalidation_counter = 0

# --- Backtesting Execution ---
if __name__ == '__main__':
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Load and preprocess data
    data = preprocess_data(filepath='data/BTC-USD-15m.csv')

    # Initialize Backtest
    bt = Backtest(
        data,
        PredictableCandleExpansion,
        cash=100000,
        commission=.002,
        exclusive_orders=True
    )

    # Run the backtest
    stats = bt.run()

    # Print the results
    print(stats)

    # Save the plot
    plot_filename = 'results/predictable_candle_expansion.html'
    bt.plot(filename=plot_filename)
    print(f"Plot saved to {plot_filename}")

    # Sanitize and save the stats to a JSON file
    sanitized_stats = sanitize_stats(stats)
    results_filename = 'results/temp_result.json'
    with open(results_filename, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)
    print(f"Results saved to {results_filename}")
