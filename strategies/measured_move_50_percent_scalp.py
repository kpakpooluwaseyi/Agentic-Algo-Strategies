import json
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from backtesting import Backtest, Strategy

def generate_synthetic_data(num_m_patterns=5, num_w_patterns=5, points_per_pattern=200):
    """
    Generates synthetic 1-minute data with embedded M and W formations.
    """
    total_points = (num_m_patterns + num_w_patterns) * points_per_pattern
    index = pd.date_range(start='2023-01-01', periods=total_points, freq='min')
    price = 100 * np.ones(total_points)

    pattern_length = points_per_pattern // 4

    for i in range(num_m_patterns):
        start_idx = i * points_per_pattern

        # M-Pattern (level drop -> retrace -> second high -> drop)
        p1 = start_idx
        p2 = p1 + pattern_length
        p3 = p2 + pattern_length
        p4 = p3 + pattern_length

        price[p1:p2] -= np.linspace(0, 5, p2 - p1) # Initial drop
        price[p2:p3] += np.linspace(0, 2.5, p3 - p2) # Retrace to 50%
        price[p3:p4] -= np.linspace(0, 3, p4 - p3) # Second drop

    for i in range(num_w_patterns):
        start_idx = (num_m_patterns + i) * points_per_pattern

        # W-Pattern (level rise -> retrace -> second low -> rise)
        p1 = start_idx
        p2 = p1 + pattern_length
        p3 = p2 + pattern_length
        p4 = p3 + pattern_length

        price[p1:p2] += np.linspace(0, 5, p2 - p1) # Initial rise
        price[p2:p3] -= np.linspace(0, 2.5, p3 - p2) # Retrace to 50%
        price[p3:p4] += np.linspace(0, 3, p4 - p3) # Second rise

    # Add noise
    noise = np.random.normal(0, 0.1, total_points)
    price += noise

    data = pd.DataFrame(index=index)
    data['Open'] = price
    data['High'] = price + np.random.uniform(0, 0.1, total_points)
    data['Low'] = price - np.random.uniform(0, 0.1, total_points)
    data['Close'] = price + np.random.normal(0, 0.05, total_points)

    return data

def preprocess_data(data_1m, prominence=1.0):
    """
    Simulates 15M analysis on 1M data to find M/W setups.
    - Resamples to 15M to find significant swing points.
    - Identifies level drops/rises.
    - Calculates 50% Fib AOI.
    - Merges this context back to the 1M data.
    """
    df_15m = data_1m['Close'].resample('15min').ohlc()

    high_peaks_idx, _ = find_peaks(df_15m['high'], prominence=prominence)
    low_peaks_idx, _ = find_peaks(-df_15m['low'], prominence=prominence)

    df_15m['swing_high'] = False
    df_15m.iloc[high_peaks_idx, df_15m.columns.get_loc('swing_high')] = True
    df_15m['swing_low'] = False
    df_15m.iloc[low_peaks_idx, df_15m.columns.get_loc('swing_low')] = True

    df_15m['setup'] = np.nan
    df_15m['aoi_level'] = np.nan
    df_15m['initial_swing_low'] = np.nan
    df_15m['initial_swing_high'] = np.nan

    last_swing_high_price = np.nan
    last_swing_low_price = np.nan

    for i in range(len(df_15m)):
        if df_15m['swing_high'].iloc[i]:
            last_swing_high_price = df_15m['high'].iloc[i]

        if df_15m['swing_low'].iloc[i]:
            # M-Formation setup identified (a high followed by a low)
            if not np.isnan(last_swing_high_price):
                aoi = last_swing_high_price - (last_swing_high_price - df_15m['low'].iloc[i]) * 0.5
                df_15m.iloc[i, df_15m.columns.get_loc('setup')] = 'short'
                df_15m.iloc[i, df_15m.columns.get_loc('aoi_level')] = aoi
                df_15m.iloc[i, df_15m.columns.get_loc('initial_swing_high')] = last_swing_high_price
                df_15m.iloc[i, df_15m.columns.get_loc('initial_swing_low')] = df_15m['low'].iloc[i]
                last_swing_high_price = np.nan # Reset to find next pattern
            last_swing_low_price = df_15m['low'].iloc[i]

        if df_15m['swing_high'].iloc[i]:
             # W-Formation setup identified (a low followed by a high)
            if not np.isnan(last_swing_low_price):
                aoi = last_swing_low_price + (df_15m['high'].iloc[i] - last_swing_low_price) * 0.5
                df_15m.iloc[i, df_15m.columns.get_loc('setup')] = 'long'
                df_15m.iloc[i, df_15m.columns.get_loc('aoi_level')] = aoi
                df_15m.iloc[i, df_15m.columns.get_loc('initial_swing_low')] = last_swing_low_price
                df_15m.iloc[i, df_15m.columns.get_loc('initial_swing_high')] = df_15m['high'].iloc[i]
                last_swing_low_price = np.nan # Reset to find next pattern
            last_swing_high_price = df_15m['high'].iloc[i]

    # Merge 15M context into 1M data
    data_1m = pd.merge(data_1m, df_15m[['setup', 'aoi_level', 'initial_swing_low', 'initial_swing_high']],
                       left_index=True, right_index=True, how='left').fillna(method='ffill')
    return data_1m.dropna()


class MeasuredMove50PercentScalpStrategy(Strategy):
    min_rr = 5
    prominence = 1.0 # This will be optimized

    def init(self):
        self.aoi_level = self.I(lambda x: x, self.data.df['aoi_level'])
        self.setup_type = self.I(lambda x: x, self.data.df['setup'].astype('category').cat.codes) # works better with codes
        self.setup_cat = self.data.df['setup'].astype('category').cat
        self.initial_swing_low = self.I(lambda x: x, self.data.df['initial_swing_low'])
        self.initial_swing_high = self.I(lambda x: x, self.data.df['initial_swing_high'])
        self.in_aoi = False

    def next(self):
        current_setup = self.setup_cat.categories[self.setup_type[-1]]
        aoi = self.aoi_level[-1]

        # Check if price is within a small range of the AOI
        in_short_aoi = current_setup == 'short' and abs(self.data.Close[-1] - aoi) < self.data.Close[-1] * 0.005
        in_long_aoi = current_setup == 'long' and abs(self.data.Close[-1] - aoi) < self.data.Close[-1] * 0.005

        if not self.position:
            # SHORT ENTRY LOGIC
            if in_short_aoi:
                # Confirmation: Bearish engulfing candle on 1M
                if self.data.Close[-1] < self.data.Open[-1] and self.data.Open[-1] > self.data.Close[-2] and self.data.Close[-1] < self.data.Open[-2]:
                    entry_price = self.data.Close[-1]
                    stop_loss = self.data.High[-1] * 1.001

                    # Target is 50% of the move from initial low to entry confirmation high
                    target = self.initial_swing_low[-1] + (self.data.High[-1] - self.initial_swing_low[-1]) * 0.5

                    risk = abs(entry_price - stop_loss)
                    reward = abs(entry_price - target)

                    if risk > 0 and reward / risk >= self.min_rr:
                        self.sell(sl=stop_loss, tp=target)

            # LONG ENTRY LOGIC
            elif in_long_aoi:
                # Confirmation: Bullish engulfing candle on 1M
                if self.data.Close[-1] > self.data.Open[-1] and self.data.Open[-1] < self.data.Close[-2] and self.data.Close[-1] > self.data.Open[-2]:
                    entry_price = self.data.Close[-1]
                    stop_loss = self.data.Low[-1] * 0.999

                    # Target is 50% of the move from initial high to entry confirmation low
                    target = self.initial_swing_high[-1] - (self.initial_swing_high[-1] - self.data.Low[-1]) * 0.5

                    risk = abs(entry_price - stop_loss)
                    reward = abs(entry_price - target)

                    if risk > 0 and reward / risk >= self.min_rr:
                        self.buy(sl=stop_loss, tp=target)

if __name__ == '__main__':
    data = generate_synthetic_data(num_m_patterns=10, num_w_patterns=10)

    # Custom optimization loop for pre-processing parameter
    best_prominence = None
    best_stats = None

    for prominence_val in np.arange(0.5, 3.0, 0.5):
        processed_data = preprocess_data(data.copy(), prominence=prominence_val)

        if processed_data.empty:
            continue

        bt = Backtest(processed_data, MeasuredMove50PercentScalpStrategy, cash=100_000, commission=.002)

        stats = bt.run(prominence=prominence_val)

        if best_stats is None or stats['Sharpe Ratio'] > best_stats['Sharpe Ratio']:
            best_stats = stats
            best_prominence = prominence_val

    print(f"Best Prominence: {best_prominence}")
    print(best_stats)

    # Run the final backtest with the best parameter to plot
    final_data = preprocess_data(data.copy(), prominence=best_prominence)
    bt = Backtest(final_data, MeasuredMove50PercentScalpStrategy, cash=100_000, commission=.002)
    final_stats = bt.run(prominence=best_prominence)


    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        # Handle cases where stats might be NaN or not present
        result_dict = {
            'strategy_name': 'measured_move_50_percent_scalp',
            'return': final_stats.get('Return [%]', 0.0),
            'sharpe': final_stats.get('Sharpe Ratio', 0.0),
            'max_drawdown': final_stats.get('Max. Drawdown [%]', 0.0),
            'win_rate': final_stats.get('Win Rate [%]', 0.0),
            'total_trades': final_stats.get('# Trades', 0)
        }
        # Convert numpy types to native python types for JSON serialization
        for key, value in result_dict.items():
            if isinstance(value, (np.number, np.bool_)):
                result_dict[key] = value.item()

        json.dump(result_dict, f, indent=2)

    print("Results saved to results/temp_result.json")

    # Generate plot
    try:
        bt.plot(filename="results/measured_move_50_percent_scalp.html")
    except Exception as e:
        print(f"Could not generate plot due to: {e}")
