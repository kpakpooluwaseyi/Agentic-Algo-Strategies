import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import json
import os
from backtesting import Strategy, Backtest

def generate_synthetic_data(periods=5000, freq='15min'):
    """
    Generates synthetic OHLC data with a clear three-day, three-level rise
    followed by a reversal, designed to trigger the strategy's short entry.
    """
    np.random.seed(42)
    start_date = '2023-01-01'
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    price = np.zeros(periods)

    # --- Create the Macro 3-Day Rise ---
    # Day 1: Rise
    day1_len = 96 # 1 day in 15min bars
    price[:day1_len] = np.linspace(100, 110, day1_len)

    # Day 2: Rise
    day2_len = 96
    price[day1_len:day1_len+day2_len] = np.linspace(110, 120, day2_len)

    # Day 3: Create the 3 Intraday Levels
    day3_start_idx = day1_len + day2_len
    day3_len = 96

    # Level 1
    level1_len = 30
    price[day3_start_idx:day3_start_idx+level1_len] = np.linspace(120, 125, level1_len)
    price[day3_start_idx+level1_len:day3_start_idx+level1_len+10] = np.linspace(125, 123, 10) # Pullback

    # Level 2
    level2_start_idx = day3_start_idx+level1_len+10
    level2_len = 30
    price[level2_start_idx:level2_start_idx+level2_len] = np.linspace(123, 128, level2_len)
    price[level2_start_idx+level2_len:level2_start_idx+level2_len+10] = np.linspace(128, 126, 10) # Pullback

    # Level 3
    level3_start_idx = level2_start_idx+level2_len+10
    level3_len = 10 # Shorter push
    price[level3_start_idx:level3_start_idx+level3_len] = np.linspace(126, 130, level3_len) # Final peak

    # Reversal candle (bearish engulfing)
    reversal_idx = level3_start_idx+level3_len
    price[reversal_idx] = 130.5 # Open higher
    price[reversal_idx+1] = 127 # Close lower

    # Subsequent drop
    price[reversal_idx+2:] = np.linspace(127, 100, periods - (reversal_idx+2))

    # --- Convert to OHLC ---
    df = pd.DataFrame(index=dates)
    noise = np.random.normal(0, 0.1, periods)
    df['Open'] = price + noise
    df['Close'] = price + np.roll(noise, 1) # Shift noise for variation
    df['High'] = np.maximum(df['Open'], df['Close']) + np.random.uniform(0, 0.2, periods)
    df['Low'] = np.minimum(df['Open'], df['Close']) - np.random.uniform(0, 0.2, periods)

    # Ensure the reversal pattern is clean
    df.loc[df.index[reversal_idx], 'High'] = 130.5
    df.loc[df.index[reversal_idx], 'Open'] = 130.2
    df.loc[df.index[reversal_idx], 'Close'] = 129.8
    df.loc[df.index[reversal_idx-1], 'Close'] = 130.0 # Previous candle close

    df.loc[df.index[reversal_idx], 'Open'] = 130.1 # Engulfing open
    df.loc[df.index[reversal_idx], 'Close'] = 129.5 # Engulfing close
    df.loc[df.index[reversal_idx], 'High'] = 130.5
    df.loc[df.index[reversal_idx], 'Low'] = 129.4


    df = df.fillna(method='ffill')
    df.columns = [c.title() for c in df.columns]

    return df

def preprocess_data(df, peak_prominence=1.5, level_prominence=0.5):
    """
    Identifies 3-day and 3-intraday level cycles based on Market Maker Method.

    Args:
        df (pd.DataFrame): Input OHLC data.
        peak_prominence (float): Prominence for detecting major daily peaks.
        level_prominence (float): Prominence for detecting intraday levels.

    Returns:
        pd.DataFrame: DataFrame with added cycle analysis columns.
    """
    # --- 1. 3-Day Cycle Detection ---
    df_daily = df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()

    # Find major peaks (PFH) and troughs (PFL)
    daily_highs = df_daily['High']
    daily_lows = df_daily['Low']
    pfh_indices, _ = find_peaks(daily_highs, prominence=peak_prominence, distance=3)
    pfl_indices, _ = find_peaks(-daily_lows, prominence=peak_prominence, distance=3)

    df_daily['cycle_day'] = 0
    df_daily['cycle_type'] = None # 'up' or 'down'

    # Process up-cycles starting from PFLs
    for pfl in pfl_indices:
        day_count = 1
        for i in range(pfl, len(df_daily) - 1):
            # Check if we hit a PFH, which ends the cycle
            if i in pfh_indices:
                break
            # Condition for a rising day
            if df_daily['Close'].iloc[i+1] > df_daily['Close'].iloc[i]:
                if df_daily['cycle_day'].iloc[i+1] == 0: # Don't overwrite existing cycle info
                    df_daily.loc[df_daily.index[i+1], 'cycle_day'] = day_count
                    df_daily.loc[df_daily.index[i+1], 'cycle_type'] = 'up'
                day_count += 1
            else:
                break # Streak broken

    # Process down-cycles starting from PFHs
    for pfh in pfh_indices:
        day_count = 1
        for i in range(pfh, len(df_daily) - 1):
            # Check if we hit a PFL, which ends the cycle
            if i in pfl_indices:
                break
            # Condition for a falling day
            if df_daily['Close'].iloc[i+1] < df_daily['Close'].iloc[i]:
                 if df_daily['cycle_day'].iloc[i+1] == 0:
                    df_daily.loc[df_daily.index[i+1], 'cycle_day'] = day_count
                    df_daily.loc[df_daily.index[i+1], 'cycle_type'] = 'down'
                 day_count += 1
            else:
                break # Streak broken

    # Map daily cycle info back to the original DataFrame
    df['date'] = df.index.date
    df_daily['date'] = df_daily.index.date
    df = pd.merge(df, df_daily[['date', 'cycle_day', 'cycle_type']], on='date', how='left')
    df.drop(columns=['date'], inplace=True)

    # --- 2. Intraday Level Detection on Day 3 ---
    df['intraday_level'] = 0
    df['peak_high'] = np.nan
    df['trough_low'] = np.nan

    # Get groups of consecutive bars that are in a 'Day 3' cycle
    day3_groups = df[df['cycle_day'] >= 3].groupby((df['cycle_day'].diff() != 0).cumsum())

    for _, group in day3_groups:
        if group.empty:
            continue

        cycle_type = group['cycle_type'].iloc[0]

        if cycle_type == 'up':
            # Find 3 levels of rise
            levels, props = find_peaks(group['High'], prominence=level_prominence, distance=5)
            if len(levels) >= 1:
                df.loc[group.index[levels[0]]:group.index[-1], 'peak_high'] = group['High'].iloc[levels[0]]
            for i, level_idx in enumerate(levels[:3]): # Max 3 levels
                df.loc[group.index[level_idx]:, 'intraday_level'] = i + 1

        elif cycle_type == 'down':
            # Find 3 levels of drop
            levels, props = find_peaks(-group['Low'], prominence=level_prominence, distance=5)
            if len(levels) >= 1:
                df.loc[group.index[levels[0]]:group.index[-1], 'trough_low'] = group['Low'].iloc[levels[0]]
            for i, level_idx in enumerate(levels[:3]): # Max 3 levels
                df.loc[group.index[level_idx]:, 'intraday_level'] = i + 1

    df['peak_high'] = df['peak_high'].ffill()
    df['trough_low'] = df['trough_low'].ffill()

    return df

class ThreeThreeReversalStrategy(Strategy):
    """
    Strategy based on the Market Maker Method's 3-day and 3-intraday
    cycle confluence for identifying major reversals.
    """
    # Optimizable parameters
    risk_reward_ratio = 2.0
    stop_loss_buffer = 1.001 # Multiplier for SL beyond peak/trough

    def init(self):
        """
        Initialize indicators from preprocessed data.
        """
        # Convert string-based cycle type to numerical data for backtesting framework
        cycle_type_numerical = self.data.df['cycle_type'].map({'up': 1, 'down': -1}).fillna(0).values

        self.cycle_day = self.I(lambda: self.data.df['cycle_day'].values, name="cycle_day")
        self.cycle_type = self.I(lambda: cycle_type_numerical, name="cycle_type_numeric")
        self.intraday_level = self.I(lambda: self.data.df['intraday_level'].values, name="intraday_level")
        self.peak_high = self.I(lambda: self.data.df['peak_high'].values, name="peak_high")
        self.trough_low = self.I(lambda: self.data.df['trough_low'].values, name="trough_low")
        self.setup_confirmed = False

    def next(self):
        """
        Entry and exit logic.
        """
        # If a position is already open, do nothing.
        if self.position:
            return

        # --- Entry Conditions ---
        is_day_3_up = self.cycle_day[-1] >= 3 and self.cycle_type[-1] == 1
        is_level_3_up = self.intraday_level[-1] == 3

        is_day_3_down = self.cycle_day[-1] >= 3 and self.cycle_type[-1] == -1
        is_level_3_down = self.intraday_level[-1] == 3

        price = self.data.Close[-1]

        # Short Entry Signal
        if is_day_3_up and is_level_3_up:
            # Look for a reversal candle (bearish engulfing or pin bar)
            is_pin_bar = (self.data.High[-1] - self.data.Close[-1]) > (self.data.Close[-1] - self.data.Open[-1]) * 2 and \
                         (self.data.Open[-1] - self.data.Low[-1]) < (self.data.Close[-1] - self.data.Open[-1])
            is_bearish_engulfing = self.data.Close[-1] < self.data.Open[-1] and \
                                   self.data.Open[-1] > self.data.Close[-2] and \
                                   self.data.Close[-1] < self.data.Open[-2]

            if is_pin_bar or is_bearish_engulfing:
                peak_high = self.peak_high[-1]
                if pd.notna(peak_high):
                    sl = peak_high * self.stop_loss_buffer
                    tp = price - (sl - price) * self.risk_reward_ratio
                    if tp < price: # Ensure TP is valid
                        self.sell(sl=sl, tp=tp)

        # Long Entry Signal
        if is_day_3_down and is_level_3_down:
            # Look for a reversal candle (bullish engulfing or pin bar)
            is_pin_bar = (self.data.Close[-1] - self.data.Low[-1]) > (self.data.Close[-1] - self.data.Open[-1]) * 2 and \
                         (self.data.High[-1] - self.data.Open[-1]) < (self.data.Close[-1] - self.data.Open[-1])
            is_bullish_engulfing = self.data.Close[-1] > self.data.Open[-1] and \
                                    self.data.Open[-1] < self.data.Close[-2] and \
                                    self.data.Close[-1] > self.data.Open[-2]

            if is_pin_bar or is_bullish_engulfing:
                trough_low = self.trough_low[-1]
                if pd.notna(trough_low):
                    sl = trough_low / self.stop_loss_buffer
                    tp = price + (price - sl) * self.risk_reward_ratio
                    if tp > price: # Ensure TP is valid
                        self.buy(sl=sl, tp=tp)

if __name__ == '__main__':
    # Configuration
    data_path = 'data/BTC-USD-15m.csv'
    output_folder = 'results'
    output_file = os.path.join(output_folder, 'temp_result.json')

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.title() for c in data.columns]
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}.")
        print("Generating synthetic data for demonstration.")
        data = generate_synthetic_data()

    # Preprocess data
    data = preprocess_data(data)

    # Initialize and run the backtest
    bt = Backtest(data, ThreeThreeReversalStrategy, cash=100_000, commission=.002)

    print("Running backtest...")
    stats = bt.run()

    # Print stats and save results
    print(stats)

    results = {
        'strategy_name': 'three_three_reversal',
        'return': stats.get('Return [%]', None),
        'sharpe': stats.get('Sharpe Ratio', None),
        'max_drawdown': stats.get('Max. Drawdown [%]', None),
        'win_rate': stats.get('Win Rate [%]', None),
        'total_trades': stats.get('# Trades', 0)
    }

    # Sanitize results for JSON serialization
    for key, value in results.items():
        if isinstance(value, (np.integer, np.floating)):
            results[key] = float(value)
        elif pd.isna(value):
            results[key] = None

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        f.write('\n') # Add a newline for POSIX compatibility

    print(f"Results saved to {output_file}")

    # Generate plot
    try:
        plot_filename = os.path.join(output_folder, 'three_three_reversal.html')
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
