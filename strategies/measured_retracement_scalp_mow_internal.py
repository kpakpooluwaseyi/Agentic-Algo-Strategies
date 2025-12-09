import pandas as pd
import numpy as np
import json
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

def generate_synthetic_data(periods=5000, m_patterns=5):
    """
    Generates synthetic 1-minute OHLC data with embedded M-formation patterns.
    """
    rng = np.random.default_rng(42)
    # Basic random walk for price
    price = 100 + np.cumsum(rng.normal(0, 0.1, periods))

    # Embed M-patterns
    pattern_period = periods // m_patterns
    for i in range(m_patterns):
        start = i * pattern_period + 200 # Add offset to avoid start
        if start + 120 > periods: continue

        # M-Pattern: Drop -> Retrace -> Drop
        # 1. Initial Drop (Level Drop)
        price[start:start+30] -= np.linspace(0, 5, 30)
        # 2. Retracement to ~50%
        price[start+30:start+60] += np.linspace(0, 2.5, 30)
        # 3. Confirmation Drop (weaker)
        price[start+60:start+90] -= np.linspace(0, 1.5, 30)
        # 4. Some noise after
        price[start+90:start+120] += np.cumsum(rng.normal(0, 0.05, 30))

    dt = pd.to_datetime(pd.date_range('2023-01-01', periods=periods, freq='1min'))
    df = pd.DataFrame({'Open': price, 'High': price, 'Low': price, 'Close': price}, index=dt)

    # Add some OHLC noise
    df['High'] += rng.uniform(0, 0.1, periods)
    df['Low'] -= rng.uniform(0, 0.1, periods)
    df['Open'] = df['Close'].shift(1)
    df.iloc[0, df.columns.get_loc('Open')] = price[0] # Set first Open

    return df.dropna()

def preprocess_data(df_1m):
    """
    Pre-processes 1M data to add 15M structure for the strategy.
    - Identifies 15M swing highs and lows.
    - Calculates the 50% Fibonacci retracement Area of Interest (AOI).
    - Makes the AOI available on every 1M bar.
    """
    # 1. Resample to 15M
    df_15m = df_1m['Close'].resample('15min').ohlc()

    # 2. Find swing points (peaks and troughs) on the 15M timeframe
    # A swing high is a peak higher than the N previous and next candles
    # A swing low is a trough lower than the N previous and next candles
    swing_range = 5 # Number of candles to look left and right
    peaks, _ = find_peaks(df_15m['high'], distance=swing_range, prominence=0.5)
    troughs, _ = find_peaks(-df_15m['low'], distance=swing_range, prominence=0.5)

    df_15m['swing_high'] = np.nan
    df_15m.iloc[peaks, df_15m.columns.get_loc('swing_high')] = df_15m.iloc[peaks]['high']
    df_15m['swing_low'] = np.nan
    df_15m.iloc[troughs, df_15m.columns.get_loc('swing_low')] = df_15m.iloc[troughs]['low']

    # 3. Identify valid 'level drops' (a swing high followed by a new swing low)
    # and calculate the Fibonacci 50% AOI in a stateful way.
    df_15m['aoi_high'] = np.nan
    df_15m['aoi_low'] = np.nan
    df_15m['structure_low'] = np.nan # Store the low that defines the structure

    last_swing_high = np.nan

    for i in range(len(df_15m)):
        current_swing_high = df_15m['swing_high'].iloc[i]
        current_swing_low = df_15m['swing_low'].iloc[i]

        # If a new swing high appears, it becomes the new reference point.
        if not np.isnan(current_swing_high):
            last_swing_high = current_swing_high

        # If a new swing low appears AND we have a reference high, a 'level drop' is confirmed.
        if not np.isnan(current_swing_low) and not np.isnan(last_swing_high):
            # Ensure the low is actually below the high
            if current_swing_low < last_swing_high:
                # Calculate the 50% Fibonacci level of this drop
                fib_level_50 = last_swing_high - (last_swing_high - current_swing_low) * 0.5

                # Define a small band around the 50% level for the AOI
                aoi_band = (last_swing_high - current_swing_low) * 0.05 # 5% of the range as the band
                df_15m.iloc[i, df_15m.columns.get_loc('aoi_high')] = fib_level_50 + aoi_band
                df_15m.iloc[i, df_15m.columns.get_loc('aoi_low')] = fib_level_50 - aoi_band
                df_15m.iloc[i, df_15m.columns.get_loc('structure_low')] = current_swing_low

                # Reset the reference high so it cannot be used again
                last_swing_high = np.nan

    # 4. Forward-fill the calculated AOI until a new one is defined
    df_15m['aoi_high'] = df_15m['aoi_high'].ffill()
    df_15m['aoi_low'] = df_15m['aoi_low'].ffill()
    df_15m['structure_low'] = df_15m['structure_low'].ffill()

    # 5. Merge the 15M context back into the 1M dataframe
    df_1m = pd.merge(df_1m, df_15m[['aoi_high', 'aoi_low', 'structure_low']], left_index=True, right_index=True, how='left')
    df_1m['aoi_high'] = df_1m['aoi_high'].ffill()
    df_1m['aoi_low'] = df_1m['aoi_low'].ffill()
    df_1m['structure_low'] = df_1m['structure_low'].ffill()
    df_1m = df_1m.dropna() # Drop rows where AOI is not yet calculated

    return df_1m

class MeasuredRetracementScalpMowInternalStrategy(Strategy):
    # --- Strategy Parameters ---
    risk_reward_ratio = 5
    confirmation_candles = 2 # Number of consecutive bearish candles for entry

    def init(self):
        # --- Pass pre-calculated data as indicators ---
        # This is the correct way to make external data available to the strategy
        self.aoi_high = self.I(lambda x: x, self.data.df['aoi_high'])
        self.aoi_low = self.I(lambda x: x, self.data.df['aoi_low'])
        self.structure_low = self.I(lambda x: x, self.data.df['structure_low'])

        # --- State Machine ---
        # State: 0 -> Searching for price to enter AOI
        # State: 1 -> Price is in AOI, waiting for 1M confirmation
        self.trade_state = 0

        # Tracks the AOI that the current setup is based on
        self.current_aoi_high = 0

        # Counts consecutive bearish candles for confirmation
        self.bearish_candle_count = 0

        # Stores the high of the confirmation pattern for SL placement
        self.confirmation_high = 0

    def next(self):
        # --- Data Access ---
        price = self.data.Close[-1]
        high = self.data.High[-1]
        # Access indicator data
        aoi_high = self.aoi_high[-1]
        aoi_low = self.aoi_low[-1]
        structure_low = self.structure_low[-1]

        # --- State Machine Logic ---

        # State 0: Searching for a setup
        if self.trade_state == 0:
            # Entry condition: Price enters the AOI from below
            if price > aoi_low and self.data.Close[-2] <= aoi_low:
                self.trade_state = 1
                self.current_aoi_high = aoi_high
                self.bearish_candle_count = 0
                self.confirmation_high = high
                # print(f"{self.data.index[-1]}: Entered AOI. State -> 1")

        # State 1: Price is in AOI, waiting for 1M confirmation
        elif self.trade_state == 1:
            # If price moves above the AOI, invalidate the setup
            if price > self.current_aoi_high:
                # print(f"{self.data.index[-1]}: Price moved above AOI. Resetting. State -> 0")
                self.trade_state = 0
                return

            # Check for bearish confirmation candles
            if self.data.Close[-1] < self.data.Open[-1]: # It's a bearish candle
                self.bearish_candle_count += 1
                self.confirmation_high = max(self.confirmation_high, high)
            else: # Not a bearish candle, reset count
                self.bearish_candle_count = 0
                self.confirmation_high = 0

            # Entry Trigger: Confirmation candle count is met
            if self.bearish_candle_count >= self.confirmation_candles:
                # print(f"{self.data.index[-1]}: Confirmation received. Attempting to sell.")
                # --- Risk Management ---
                stop_loss = self.confirmation_high + 0.1 # Place SL above the confirmation high

                # TP is the 50% measured move of the center peak
                # Center peak high is our entry, low is the structure low
                entry_price = price
                take_profit = entry_price - (entry_price - structure_low) * 0.5

                # --- R:R Check ---
                risk = stop_loss - entry_price
                reward = entry_price - take_profit

                if risk > 0 and reward / risk >= self.risk_reward_ratio:
                    self.sell(sl=stop_loss, tp=take_profit, size=1)

                # Reset state machine regardless of trade placement
                self.trade_state = 0
                self.bearish_candle_count = 0
                self.confirmation_high = 0

if __name__ == '__main__':
    # 1. Generate and preprocess data
    data_1m = generate_synthetic_data(periods=20000) # Use a larger dataset for more robust optimization
    processed_data = preprocess_data(data_1m.copy())

    # 2. Initialize the backtest
    bt = Backtest(processed_data, MeasuredRetracementScalpMowInternalStrategy,
                  cash=100_000, commission=.002)

    # 3. Optimize the strategy
    # Find the best combination of risk/reward and confirmation candles
    stats = bt.optimize(
        risk_reward_ratio=range(3, 8, 1),
        confirmation_candles=range(2, 4, 1),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.risk_reward_ratio > 2 # Example constraint
    )

    print("Best stats:")
    print(stats)

    # 4. Save the results to a JSON file
    import os
    try:
        os.makedirs('results', exist_ok=True)

        # Ensure results are JSON serializable
        results_dict = {
            'strategy_name': 'measured_retracement_scalp_mow_internal',
            'return': stats.get('Return [%]', 0.0),
            'sharpe': stats.get('Sharpe Ratio', 0.0),
            'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
            'win_rate': stats.get('Win Rate [%]', 0.0),
            'total_trades': stats.get('# Trades', 0)
        }

        # Convert numpy types to native Python types
        for key, value in results_dict.items():
            if isinstance(value, (np.int64, np.int32)):
                results_dict[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                results_dict[key] = float(value)

        # Handle NaN for sharpe ratio if no trades
        if results_dict['total_trades'] == 0:
            results_dict['sharpe'] = None

        with open('results/temp_result.json', 'w') as f:
            json.dump(results_dict, f, indent=2)

        print("\nResults saved to results/temp_result.json")

    except Exception as e:
        print(f"\nError saving results: {e}")

    # 5. Generate the performance plot
    try:
        plot_filename = 'results/measured_retracement_scalp_mow_internal.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error generating plot: {e}")
