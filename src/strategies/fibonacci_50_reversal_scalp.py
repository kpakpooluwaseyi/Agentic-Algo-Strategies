
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks

def generate_synthetic_data(n_patterns=10):
    """
    Generates synthetic price data with distinct 'M' and 'W' patterns.
    Each pattern consists of a clear swing, a retracement to the 50% level,
    a reversal, and then a move to a target level.
    """
    base_price = 100
    pattern_length = 200  # Bars per pattern
    total_bars = n_patterns * pattern_length

    rng = np.random.default_rng(42)

    timestamps = pd.to_datetime(pd.date_range('2023-01-01', periods=total_bars, freq='min'))
    price_changes = rng.normal(0, 0.05, size=total_bars)
    price = base_price + np.cumsum(price_changes)

    # Inject M/W patterns
    for i in range(n_patterns):
        start_idx = i * pattern_length + 20 # Add buffer

        # Determine pattern type
        is_m_pattern = rng.choice([True, False])

        if is_m_pattern: # Create an 'M' pattern (for a SHORT trade)
            # 1. Initial drop (A to B)
            p_a = price[start_idx]
            p_b = p_a - rng.uniform(2, 4)
            price[start_idx+10:start_idx+30] = np.linspace(p_a, p_b, 20)

            # 2. Retracement to 50% (B to C)
            p_c = p_b + (p_a - p_b) * 0.5
            price[start_idx+30:start_idx+50] = np.linspace(p_b, p_c, 20)

            # 3. Reversal from 50% (C), now with 1-2-3 pattern
            p_point2 = p_c - rng.uniform(0.3, 0.5)
            price[start_idx+50:start_idx+60] = np.linspace(p_c, p_point2, 10)
            p_point3 = p_point2 + rng.uniform(0.1, 0.2)
            price[start_idx+60:start_idx+70] = np.linspace(p_point2, p_point3, 10)
            p_d_target = p_c - (p_c - p_b) * 0.5
            price[start_idx+70:start_idx+100] = np.linspace(p_point3, p_d_target, 30)

            # Fill the rest with noise
            remaining_length = pattern_length - (start_idx + 100 - i * pattern_length)
            if remaining_length > 0:
                price[start_idx+100:(i+1)*pattern_length] = p_d_target + np.cumsum(rng.normal(0, 0.05, size=remaining_length))

        else: # Create a 'W' pattern (for a LONG trade)
            # 1. Initial rally (A to B)
            p_a = price[start_idx]
            p_b = p_a + rng.uniform(2, 4)
            price[start_idx+10:start_idx+30] = np.linspace(p_a, p_b, 20)

            # 2. Retracement to 50% (B to C)
            p_c = p_b - (p_b - p_a) * 0.5
            price[start_idx+30:start_idx+50] = np.linspace(p_b, p_c, 20)

            # 3. Reversal from 50% (C), now with 1-2-3 pattern
            p_point2 = p_c + rng.uniform(0.3, 0.5)
            price[start_idx+50:start_idx+60] = np.linspace(p_c, p_point2, 10)
            p_point3 = p_point2 - rng.uniform(0.1, 0.2)
            price[start_idx+60:start_idx+70] = np.linspace(p_point2, p_point3, 10)
            p_d_target = p_c + (p_b - p_c) * 0.5
            price[start_idx+70:start_idx+100] = np.linspace(p_point3, p_d_target, 30)

            # Fill the rest with noise
            remaining_length = pattern_length - (start_idx + 100 - i * pattern_length)
            if remaining_length > 0:
                price[start_idx+100:(i+1)*pattern_length] = p_d_target + np.cumsum(rng.normal(0, 0.05, size=remaining_length))

    # Create DataFrame
    data = pd.DataFrame({
        'Open': price,
        'High': price + rng.uniform(0, 0.1, size=total_bars),
        'Low': price - rng.uniform(0, 0.1, size=total_bars),
        'Close': price,
        'Volume': rng.uniform(100, 1000, size=total_bars)
    }, index=timestamps)

    # Ensure OHLC consistency
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


class Fibonacci50ReversalScalpStrategy(Strategy):
    # Optimization parameters
    reversal_confirmation_bars = 5
    swing_threshold = 10

    # State variables
    entry_price = 0

    def init(self):
        # Pre-processing logic is now inside the strategy
        data_1m = self.data.df

        # More accurate resampling
        resample_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        data_15m = data_1m.resample('15min').apply(resample_rules).dropna()

        # Find swing highs and lows on 15M data
        peak_indices, _ = find_peaks(data_15m['High'], distance=self.swing_threshold)
        trough_indices, _ = find_peaks(-data_15m['Low'], distance=self.swing_threshold)

        swing_highs = data_15m.iloc[peak_indices].assign(type='high')
        swing_lows = data_15m.iloc[trough_indices].assign(type='low')

        swings = pd.concat([swing_highs, swing_lows]).sort_index()

        # Initialize columns for setup signals
        num_bars = len(data_1m)
        setup_columns = {
            'is_short_setup': [False] * num_bars,
            'is_long_setup': [False] * num_bars,
            'aoi_level': [np.nan] * num_bars,
            'initial_swing_high': [np.nan] * num_bars,
            'initial_swing_low': [np.nan] * num_bars,
            'center_peak': [np.nan] * num_bars,
            'center_trough': [np.nan] * num_bars
        }

        if len(swings) >= 3:
            for i in range(len(swings) - 2):
                s1, s2, s3 = swings.iloc[i], swings.iloc[i+1], swings.iloc[i+2]

                setup_info = None
                # M-Pattern (Short Setup)
                if s1['type'] == 'high' and s2['type'] == 'low' and s3['type'] == 'high':
                    initial_high, initial_low, center_peak = s1['High'], s2['Low'], s3['High']
                    if initial_high > initial_low and center_peak > initial_low:
                        fib_50 = initial_high - (initial_high - initial_low) * 0.5
                        if abs(center_peak - fib_50) < (initial_high - initial_low) * 0.1:
                            setup_info = ('short', fib_50, initial_high, initial_low, center_peak, np.nan)

                # W-Pattern (Long Setup)
                elif s1['type'] == 'low' and s2['type'] == 'high' and s3['type'] == 'low':
                    initial_low, initial_high, center_trough = s1['Low'], s2['High'], s3['Low']
                    if initial_low < initial_high and center_trough < initial_high:
                        fib_50 = initial_low + (initial_high - initial_low) * 0.5
                        if abs(center_trough - fib_50) < (initial_high - initial_low) * 0.1:
                             setup_info = ('long', fib_50, initial_high, initial_low, np.nan, center_trough)

                if setup_info:
                    setup_type, aoi, i_high, i_low, c_peak, c_trough = setup_info
                    start_time = s2.name
                    end_time = swings.index[i+2] if i + 3 < len(swings) else data_1m.index[-1]

                    indices = np.where((data_1m.index >= start_time) & (data_1m.index < end_time))[0]
                    for idx in indices:
                        if setup_type == 'short':
                            setup_columns['is_short_setup'][idx] = True
                        else:
                            setup_columns['is_long_setup'][idx] = True
                        setup_columns['aoi_level'][idx] = aoi
                        setup_columns['initial_swing_high'][idx] = i_high
                        setup_columns['initial_swing_low'][idx] = i_low
                        setup_columns['center_peak'][idx] = c_peak
                        setup_columns['center_trough'][idx] = c_trough

        # Assign the calculated columns to the main dataframe
        for col_name, col_values in setup_columns.items():
            self.data.df[col_name] = col_values

        # State tracking for 1-2-3 reversal pattern
        self.point1 = None
        self.point2 = None
        self.point3 = None
        self.setup_type = None # 'long' or 'short'

    def next(self):
        # Avoid acting on old data if a setup has just expired
        if not self.data.is_long_setup[-1] and not self.data.is_short_setup[-1]:
            self.point1 = self.point2 = self.point3 = self.setup_type = None
            return

        # ==================
        # SHORT (M-Pattern)
        # ==================
        if self.data.is_short_setup[-1] and not self.position and not self.setup_type:
            # We have a 15M setup. Start looking for a 1M 1-2-3 reversal.
            # Point 1 is the high of the AOI rejection (center peak)
            self.point1 = self.data.center_peak[-1]
            self.setup_type = 'short'

        if self.setup_type == 'short':
            # Look for Point 2: a swing low after the peak
            if self.point1 and not self.point2:
                # Simple check: has the price dropped for a few bars?
                if self.data.Close[-1] < self.data.Close[-self.reversal_confirmation_bars]:
                    self.point2 = self.data.Low[-self.reversal_confirmation_bars:].min()

            # Look for Point 3: a lower high after Point 2
            if self.point2 and not self.point3:
                if self.data.High[-1] < self.point1 and self.data.Close[-1] > self.point2:
                     # Simple check: has the price risen a bit?
                     if self.data.Close[-1] > self.data.Close[-self.reversal_confirmation_bars]:
                         self.point3 = self.data.High[-1] # Tentative point 3

            # ENTRY condition: Price breaks below Point 2
            if self.point1 and self.point2 and self.point3:
                if self.data.Close[-1] < self.point2:
                    sl = self.point1
                    # Target (Fib 2 from initial low to center peak)
                    tp = self.point1 - (self.point1 - self.data.initial_swing_low[-1]) * 0.5

                    if tp < self.data.Close[-1]: # Ensure TP is valid
                        self.sell(sl=sl, tp=tp)
                        self.point1 = self.point2 = self.point3 = self.setup_type = None

        # Reset state if a trade was closed on this bar
        if len(self.trades) < len(self.closed_trades):
             self.point1 = self.point2 = self.point3 = self.setup_type = None


        # ==================
        # LONG (W-Pattern)
        # ==================
        if self.data.is_long_setup[-1] and not self.position and not self.setup_type:
            # Point 1 is the low of the AOI rejection (center trough)
            self.point1 = self.data.center_trough[-1]
            self.setup_type = 'long'

        if self.setup_type == 'long':
            # Look for Point 2: a swing high after the trough
            if self.point1 and not self.point2:
                if self.data.Close[-1] > self.data.Close[-self.reversal_confirmation_bars]:
                    self.point2 = self.data.High[-self.reversal_confirmation_bars:].max()

            # Look for Point 3: a higher low after Point 2
            if self.point2 and not self.point3:
                if self.data.Low[-1] > self.point1 and self.data.Close[-1] < self.point2:
                    if self.data.Close[-1] < self.data.Close[-self.reversal_confirmation_bars]:
                        self.point3 = self.data.Low[-1] # Tentative point 3

            # ENTRY condition: Price breaks above Point 2
            if self.point1 and self.point2 and self.point3:
                if self.data.Close[-1] > self.point2:
                    sl = self.point1
                    # Target (Fib 2 from initial high to center trough)
                    tp = self.point1 + (self.data.initial_swing_high[-1] - self.point1) * 0.5

                    if tp > self.data.Close[-1]: # Ensure TP is valid
                        self.buy(sl=sl, tp=tp)
                        self.point1 = self.point2 = self.point3 = self.setup_type = None

        # Reset state if a trade was closed on this bar
        if len(self.trades) < len(self.closed_trades):
            self.point1 = self.point2 = self.point3 = self.setup_type = None

if __name__ == '__main__':
    # Using a single set of parameters for the initial data processing run.
    # The strategy will then be optimized over its parameters.
    data_1m = generate_synthetic_data(n_patterns=50)

    # Note: When optimizing a parameter that affects preprocessing,
    # the ideal approach is to re-run preprocessing for each optimization iteration.
    # `backtesting.py` doesn't support this directly. A workaround is to pass
    # a callable to `Backtest` that regenerates data, but for this case,
    # we'll process with a median value and optimize the strategy logic parameters.

    bt = Backtest(data_1m, Fibonacci50ReversalScalpStrategy, cash=100000, commission=.002)

    # Optimize the strategy
    stats = bt.optimize(
        reversal_confirmation_bars=range(3, 10, 1),
        swing_threshold=range(5, 15, 2),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.reversal_confirmation_bars > 2 and p.swing_threshold > 3
    )

    print("Best stats found:")
    print(stats)

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Handle cases where no trades were made
    win_rate = stats.get('Win Rate [%]', 0)
    sharpe = stats.get('Sharpe Ratio', 0)

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'fibonacci_50_reversal_scalp',
            'return': float(stats['Return [%]']),
            'sharpe': float(sharpe),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(win_rate),
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    # Generate plot
    bt.plot(filename="results/fibonacci_50_reversal_scalp.html")
