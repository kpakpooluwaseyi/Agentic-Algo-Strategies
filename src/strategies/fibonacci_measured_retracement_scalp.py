
import json
import pandas as pd
import numpy as np

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG

# --- Custom Indicator Functions ---
# backtesting.py requires indicator functions to be defined at the module level.

def highest(arr: np.ndarray, n: int) -> np.ndarray:
    """Returns the highest value over the last n periods."""
    # Pad with NaNs for the initial n-1 values
    padded_arr = np.concatenate([np.full(n - 1, np.nan), arr])
    # Use a rolling window to find the max
    rolling_max = pd.Series(padded_arr).rolling(window=n).max().to_numpy()
    return rolling_max[n - 1:]

def lowest(arr: np.ndarray, n: int) -> np.ndarray:
    """Returns the lowest value over the last n periods."""
    # Pad with NaNs for the initial n-1 values
    padded_arr = np.concatenate([np.full(n - 1, np.nan), arr])
    # Use a rolling window to find the min
    rolling_min = pd.Series(padded_arr).rolling(window=n).min().to_numpy()
    return rolling_min[n - 1:]

class FibonacciMeasuredRetracementScalpStrategy(Strategy):
    """
    Implements a trading strategy based on Fibonacci retracements after a
    significant price move, approximating a 15M/1M multi-timeframe logic.

    Entry Rules (Short):
    1. Identify a significant price drop (Level Drop) over `swing_lookback` periods.
    2. Calculate the 50% Fibonacci retracement of this drop (Area of Interest - AOI).
    3. When price enters the AOI, look for signs of weakness (e.g., a failure to
       make a new high over `confirmation_lookback` periods).
    4. Enter a SHORT position.

    Exit Rules (Short):
    1. Calculate the 50% retracement of the range from the Level Drop's low to the
       retracement's high (Center Peak).
    2. Place Take Profit at this level.

    Risk Management (Short):
    1. Place Stop Loss just above the high of the Center Peak.
    """
    # --- Optimization Parameters ---
    # Lookback period to identify the initial significant price swing (Level Drop/Rise)
    swing_lookback = 20

    # Lookback period to confirm weakness/strength within the AOI
    confirmation_lookback = 5

    def init(self):
        """
        Initialize indicators and strategy state.
        """
        # --- Indicators ---
        # self.I() is used to create and manage indicators.
        # We need to identify historical peaks and troughs to define our swings.
        self.highest_high = self.I(highest, self.data.High, self.swing_lookback, name="Highest High")
        self.lowest_low = self.I(lowest, self.data.Low, self.swing_lookback, name="Lowest Low")

        # --- State Variables ---
        # These variables help track the state of the current trade setup.
        # We use dictionaries to track state for both Long and Short setups independently.
        self.setup_state = {
            'long': {'peak': None, 'trough': None, 'aoi': None, 'center_trough': None},
            'short': {'peak': None, 'trough': None, 'aoi': None, 'center_peak': None}
        }


    def next(self):
        """
        The main strategy logic, executed on each bar of the data.
        """
        # --- Aliases for easier access ---
        price = self.data.Close[-1]

        # --- Prevent entries if a position is already open ---
        if self.position:
            return

        # --- SHORT Trade Logic ---

        # 1. Identify Level Drop: Look for a significant new low.
        # `self.lowest_low[-2]` refers to the lowest low of the period *before* the last bar.
        # `price < self.lowest_low[-2]` checks if the current price just broke that low.
        # A simpler approach is to check if the swing lookback period just produced a new low.
        is_new_low = self.data.Low[-1] == self.lowest_low[-1]

        if is_new_low:
            # A new significant low has been identified. This is our "Level Drop".
            # The peak is the highest point during this drop.
            self.setup_state['short']['peak'] = self.highest_high[-1]
            self.setup_state['short']['trough'] = self.lowest_low[-1]

            # 2. Calculate Area of Interest (AOI): 50% Fibonacci Retracement
            level_drop_range = self.setup_state['short']['peak'] - self.setup_state['short']['trough']
            self.setup_state['short']['aoi'] = self.setup_state['short']['peak'] - (level_drop_range * 0.5)

        # 3. Entry Logic: Check if price has entered the AOI and shows weakness.
        if self.setup_state['short']['aoi'] is not None and price >= self.setup_state['short']['aoi']:
            # Price has entered the Area of Interest.
            # Now, we look for "weakness". We approximate this by checking if the price
            # fails to make a new high over a shorter `confirmation_lookback` period.
            recent_high = highest(self.data.High, self.confirmation_lookback)[-1]

            # If the current high is the highest in the confirmation window, we assume strength.
            # We wait for a bar that *doesn't* make a new high.
            if self.data.High[-1] < recent_high:
                # This is our signal to enter.
                center_peak = recent_high
                trough = self.setup_state['short']['trough']

                # 4. Risk and Exit Calculation
                sl = center_peak * 1.001 # Stop loss just above the center peak

                center_peak_range = center_peak - trough
                tp = center_peak - (center_peak_range * 0.5) # TP at 50% of the center peak range

                # Ensure TP is valid (below entry)
                if tp < price:
                    self.sell(sl=sl, tp=tp)

                # Reset state after attempting a trade
                self.setup_state['short'] = {'peak': None, 'trough': None, 'aoi': None, 'center_peak': None}


        # --- LONG Trade Logic (Inverted) ---

        # 1. Identify Level Rise: Look for a significant new high.
        is_new_high = self.data.High[-1] == self.highest_high[-1]

        if is_new_high:
            # A new significant high has been identified.
            self.setup_state['long']['peak'] = self.highest_high[-1]
            self.setup_state['long']['trough'] = self.lowest_low[-1]

            # 2. Calculate Area of Interest (AOI): 50% Fibonacci Retracement
            level_rise_range = self.setup_state['long']['peak'] - self.setup_state['long']['trough']
            self.setup_state['long']['aoi'] = self.setup_state['long']['trough'] + (level_rise_range * 0.5)

        # 3. Entry Logic: Check if price has entered the AOI and shows strength.
        if self.setup_state['long']['aoi'] is not None and price <= self.setup_state['long']['aoi']:
            # Price has entered the Area of Interest.
            # Look for "strength" by checking if price fails to make a new low.
            recent_low = lowest(self.data.Low, self.confirmation_lookback)[-1]

            if self.data.Low[-1] > recent_low:
                # This is our signal to enter.
                center_trough = recent_low
                peak = self.setup_state['long']['peak']

                # 4. Risk and Exit Calculation
                sl = center_trough * 0.999 # Stop loss just below the center trough

                center_trough_range = peak - center_trough
                tp = center_trough + (center_trough_range * 0.5) # TP at 50% of the center trough range

                # Ensure TP is valid (above entry)
                if tp > price:
                    self.buy(sl=sl, tp=tp)

                # Reset state after attempting a trade
                self.setup_state['long'] = {'peak': None, 'trough': None, 'aoi': None, 'center_trough': None}


if __name__ == '__main__':
    # --- Data Loading ---
    # Use a sample dataset provided by the library.
    # In a real-world scenario, you would load your own CSV data.
    # The GOOG data is daily, which is not ideal for this strategy's logic,
    # but it serves for demonstrating the implementation. For a real test,
    # 1-minute or 15-minute data would be required.
    data = GOOG.copy()
    data = data.iloc[-2000:] # Use a smaller subset for faster optimization

    # --- Backtest Initialization ---
    bt = Backtest(
        data,
        FibonacciMeasuredRetracementScalpStrategy,
        cash=100000,
        commission=.002,
        trade_on_close=True,
        exclusive_orders=True
    )

    # --- Optimization ---
    # Find the best parameters for the strategy.
    stats = bt.optimize(
        swing_lookback=range(10, 50, 5),
        confirmation_lookback=range(3, 15, 2),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.swing_lookback > p.confirmation_lookback # Ensure lookbacks are logical
    )

    print("Best optimization stats:")
    print(stats)
    print("Best parameters:")
    print(stats._strategy)

    # --- Output Results ---
    # Create the 'results' directory if it doesn't exist.
    import os
    os.makedirs('results', exist_ok=True)

    # Save the backtest results to a JSON file.
    # We cast stats to native Python types to ensure JSON compatibility.
    if stats['# Trades'] > 0:
        with open('results/temp_result.json', 'w') as f:
            json.dump({
                'strategy_name': 'fibonacci_measured_retracement_scalp',
                'return': float(stats['Return [%]']),
                'sharpe': float(stats['Sharpe Ratio']),
                'max_drawdown': float(stats['Max. Drawdown [%]']),
                'win_rate': float(stats['Win Rate [%]']),
                'total_trades': int(stats['# Trades'])
            }, f, indent=2)
    else:
        # Handle the case with no trades
        with open('results/temp_result.json', 'w') as f:
            json.dump({
                'strategy_name': 'fibonacci_measured_retracement_scalp',
                'return': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }, f, indent=2)


    # --- Plotting ---
    # Generate an interactive plot of the backtest.
    bt.plot(filename='fibonacci_retracement_backtest.html', open_browser=False)
