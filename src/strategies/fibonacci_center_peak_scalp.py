
import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

class FibonacciCenterPeakScalpStrategy(Strategy):
    """
    Strategy based on identifying M/W patterns using Fibonacci retracements.
    15M Analysis: Identify a significant price move (Leg 1) and a 50% retracement level (AOI).
    1M Execution: Enter on a reversal pattern at the AOI (Center Peak) and target the 50%
                  retracement of the Center Peak move.
    """

    # Optimization parameters
    rr_ratio = 5  # Risk-to-Reward Ratio

    def init(self):
        """
        Initialize the strategy. Pre-calculated data columns will be accessed here.
        """
        # self.I() is not needed for pre-calculated columns accessed directly.
        # The `pattern_type` column caused errors as it's a string.
        pass

    def next(self):
        """
        Main strategy logic executed on each 1M candle.
        """
        # Access pre-processed data directly from the DataFrame to avoid errors.
        current_aoi_high = self.data.df.get('aoi_high', pd.Series(np.nan)).iloc[-1]
        current_aoi_low = self.data.df.get('aoi_low', pd.Series(np.nan)).iloc[-1]
        current_pattern_type = self.data.df.get('pattern_type', pd.Series(None)).iloc[-1]
        leg1_start_price = self.data.df.get('leg1_start_price', pd.Series(np.nan)).iloc[-1]

        # Ensure we have enough data
        if len(self.data) < 5 or np.isnan(leg1_start_price):
            return

        price = self.data.Close[-1]

        # ===================
        # ENTRY LOGIC
        # ===================
        if not self.position:
            # M-Pattern (Short) Entry
            # 1. Price is inside the 15M AOI.
            # 2. A bearish reversal pattern forms (e.g., swing high followed by a candle closing below its low).
            if current_pattern_type == 'M' and current_aoi_low < price < current_aoi_high:
                # Reversal: A peak formed at candle [-2], and candle [-1] closed below its low.
                if self.data.Close[-1] < self.data.Low[-2]:
                    center_peak_high = self.data.High[-2]
                    stop_loss = center_peak_high * 1.001

                    # Correct TP calc: 50% retracement from Leg 1 start to center peak
                    take_profit = center_peak_high - 0.5 * (center_peak_high - leg1_start_price)

                    # Basic validation and RR check
                    if take_profit < price and (price - take_profit) >= self.rr_ratio * (stop_loss - price):
                        self.sell(sl=stop_loss, tp=take_profit)

            # W-Pattern (Long) Entry
            # 1. Price is inside the 15M AOI.
            # 2. A bullish reversal pattern forms (e.g., swing low followed by a candle closing above its high).
            if current_pattern_type == 'W' and current_aoi_low < price < current_aoi_high:
                # Reversal: A trough at candle [-2], and candle [-1] closed above its high.
                if self.data.Close[-1] > self.data.High[-2]:
                    center_peak_low = self.data.Low[-2]
                    stop_loss = center_peak_low * 0.999

                    # Correct TP calc: 50% retracement from Leg 1 start to center peak
                    take_profit = center_peak_low + 0.5 * (leg1_start_price - center_peak_low)

                    # Basic validation and RR check
                    if take_profit > price and (take_profit - price) >= self.rr_ratio * (price - stop_loss):
                        self.buy(sl=stop_loss, tp=take_profit)


def generate_synthetic_data(n_patterns=10, points_per_minute=1):
    """
    Generates synthetic 1-minute OHLC data with clear M and W patterns.
    """
    data = []
    start_price = 100
    current_time = pd.to_datetime('2023-01-01')

    for i in range(n_patterns):
        # Determine pattern type
        pattern_type = 'M' if i % 2 == 0 else 'W'

        # --- Leg 1 (15M significant move) ---
        leg1_duration = np.random.randint(60, 120) * points_per_minute
        leg1_end_price = start_price + \
            np.random.uniform(5, 10) * (1 if pattern_type == 'M' else -1)
        leg1_prices = np.linspace(start_price, leg1_end_price, leg1_duration)

        # --- Retracement to AOI ---
        retracement_duration = np.random.randint(45, 90) * points_per_minute
        aoi_price = start_price + 0.5 * (leg1_end_price - start_price)
        retracement_prices = np.linspace(
            leg1_end_price, aoi_price, retracement_duration)

        # --- Center Peak Formation (1M reversal) ---
        center_peak_duration = np.random.randint(10, 20) * points_per_minute
        center_peak_price = aoi_price + \
            np.random.uniform(0.5, 1) * (1 if pattern_type == 'M' else -1)
        peak_prices1 = np.linspace(
            aoi_price, center_peak_price, center_peak_duration // 2)
        peak_prices2 = np.linspace(
            center_peak_price, aoi_price, center_peak_duration - (center_peak_duration // 2))

        # --- Leg 2 (Continuation) ---
        leg2_duration = np.random.randint(60, 120) * points_per_minute
        leg2_end_price = start_price  # Return to base
        leg2_prices = np.linspace(aoi_price, leg2_end_price, leg2_duration)

        # Combine all parts
        full_pattern_prices = np.concatenate(
            [leg1_prices, retracement_prices, peak_prices1, peak_prices2, leg2_prices])

        # Create DataFrame
        for price in full_pattern_prices:
            # Add some noise for OHLC
            open_p = price + np.random.normal(0, 0.05)
            high_p = max(open_p, price) + np.random.uniform(0, 0.1)
            low_p = min(open_p, price) - np.random.uniform(0, 0.1)
            close_p = price + np.random.normal(0, 0.05)
            data.append([current_time, open_p, high_p, low_p, close_p])
            current_time += pd.Timedelta(minutes=1)

        start_price = leg2_end_price # Start next pattern from the end of the last one

    df = pd.DataFrame(
        data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close'])
    df.set_index('Timestamp', inplace=True)
    return df


def preprocess_data(df_1m):
    """
    Pre-processes 1M data to identify 15M setup conditions.
    1. Resample to 15M.
    2. Identify swing points (peaks/troughs) to define Leg 1.
    3. Calculate 50% Fibonacci AOI.
    4. Merge this information back into the 1M DataFrame.
    """
    # Correctly resample the entire DataFrame to preserve OHLC columns
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    df_15m['pattern_type'] = None
    df_15m['aoi_high'] = np.nan
    df_15m['aoi_low'] = np.nan
    df_15m['leg1_start_price'] = np.nan

    # More robust swing detection using rolling windows
    rolling_window = 15 # A larger window to find more significant moves
    df_15m['is_swing_high'] = df_15m['High'] == df_15m['High'].rolling(rolling_window, center=True).max()
    df_15m['is_swing_low'] = df_15m['Low'] == df_15m['Low'].rolling(rolling_window, center=True).min()

    swing_highs = df_15m[df_15m['is_swing_high']]
    swing_lows = df_15m[df_15m['is_swing_low']]

    # Iterate through swings to define patterns and AOIs
    last_swing_idx = None
    last_swing_type = None

    for idx, row in df_15m.iterrows():
        is_high = row['is_swing_high']
        is_low = row['is_swing_low']

        if is_high:
            if last_swing_type == 'low':
                # Found a low followed by a high: Potential M-Pattern Leg 1
                leg1_start_val = df_15m.loc[last_swing_idx, 'Low']
                leg1_end_val = row['High']
                aoi = leg1_start_val + 0.5 * (leg1_end_val - leg1_start_val)

                # Mark the AOI only for the period until the next swing
                df_15m.loc[last_swing_idx:idx, 'pattern_type'] = 'M'
                df_15m.loc[last_swing_idx:idx, 'aoi_low'] = aoi * 0.995
                df_15m.loc[last_swing_idx:idx, 'aoi_high'] = aoi * 1.005
                df_15m.loc[last_swing_idx:idx, 'leg1_start_price'] = leg1_start_val

            last_swing_type = 'high'
            last_swing_idx = idx

        elif is_low:
            if last_swing_type == 'high':
                # Found a high followed by a low: Potential W-Pattern Leg 1
                leg1_start_val = df_15m.loc[last_swing_idx, 'High']
                leg1_end_val = row['Low']
                aoi = leg1_start_val - 0.5 * (leg1_start_val - leg1_end_val)

                # Mark the AOI only for the period until the next swing
                df_15m.loc[last_swing_idx:idx, 'pattern_type'] = 'W'
                df_15m.loc[last_swing_idx:idx, 'aoi_low'] = aoi * 0.995
                df_15m.loc[last_swing_idx:idx, 'aoi_high'] = aoi * 1.005
                df_15m.loc[last_swing_idx:idx, 'leg1_start_price'] = leg1_start_val

            last_swing_type = 'low'
            last_swing_idx = idx

    # Merge AOI info back to 1M data
    df_1m_processed = pd.merge_asof(
        df_1m, df_15m[['pattern_type', 'aoi_high', 'aoi_low', 'leg1_start_price']],
        left_index=True, right_index=True,
        direction='backward'
    )
    # Address FutureWarning by using .ffill() directly
    df_1m_processed = df_1m_processed.ffill()
    return df_1m_processed


if __name__ == '__main__':
    # 1. Generate or Load Data
    data_1m = generate_synthetic_data(n_patterns=20)

    # 2. Pre-process Data for MTF Analysis
    try:
        data_processed = preprocess_data(data_1m.copy())
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        # Add columns manually if preprocessing fails, for debugging
        data_processed = data_1m
        data_processed['aoi_high'] = np.nan
        data_processed['aoi_low'] = np.nan
        data_processed['pattern_type'] = None


    # 3. Run Backtest & Optimization
    bt = Backtest(data_processed, FibonacciCenterPeakScalpStrategy,
                  cash=100_000, commission=.002)

    print("Running optimization...")
    stats = bt.optimize(
        rr_ratio=range(3, 8, 1),
        maximize='Sharpe Ratio',
        max_tries=50  # Reduced for faster execution
    )

    print("Best stats:", stats)

    # 4. Save Results
    import os
    os.makedirs('results', exist_ok=True)

    # Handle cases with no trades
    win_rate = stats.get('Win Rate [%]', 0)
    sharpe = stats.get('Sharpe Ratio', 0)

    # Ensure types are JSON serializable
    results_dict = {
        'strategy_name': 'fibonacci_center_peak_scalp',
        'return': float(stats.get('Return [%]', 0)),
        'sharpe': sharpe if np.isfinite(sharpe) else None,
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)),
        'win_rate': win_rate if np.isfinite(win_rate) else None,
        'total_trades': int(stats.get('# Trades', 0))
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("Results saved to results/temp_result.json")

    # 5. Generate Plot
    try:
        bt.plot()
    except Exception as e:
        print(f"Could not generate plot: {e}")
