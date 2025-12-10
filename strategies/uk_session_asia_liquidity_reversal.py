
import json
import os
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# --- Configuration ---
ASIA_START_HOUR = 0  # 00:00 UTC
ASIA_END_HOUR = 8    # 08:00 UTC
UK_START_HOUR = 7    # 07:00 UTC
UK_END_HOUR = 16     # 16:00 UTC

# --- Synthetic Data Generation ---
def generate_synthetic_data():
    """Generates synthetic 15-minute data with specific patterns for the strategy."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='15min')
    price = 1.1000
    data = []

    for date in dates:
        # Asia Session (Quiet Range)
        if ASIA_START_HOUR <= date.hour < ASIA_END_HOUR:
            change = np.random.uniform(-0.0005, 0.0005)
        # UK Session (Potential for Spike)
        elif UK_START_HOUR <= date.hour < UK_END_HOUR:
            change = np.random.uniform(-0.0015, 0.0015)
        # Other times (standard volatility)
        else:
            change = np.random.uniform(-0.0010, 0.0010)

        price += change
        data.append({'Open': price, 'High': price + np.random.uniform(0, 0.0005),
                     'Low': price - np.random.uniform(0, 0.0005), 'Close': price + np.random.uniform(-0.0003, 0.0003)})

    df = pd.DataFrame(data, index=dates)
    df['Volume'] = np.random.randint(100, 1000, size=len(df))

    # --- Inject a textbook short pattern ---
    pattern_day_short = '2023-01-10'
    # Asia Range
    df.loc[f'{pattern_day_short} 00:00:00':f'{pattern_day_short} 07:45:00', ['Open', 'Close']] = 1.1000
    df.loc[f'{pattern_day_short} 00:00:00':f'{pattern_day_short} 07:45:00', 'High'] = 1.1010
    df.loc[f'{pattern_day_short} 00:00:00':f'{pattern_day_short} 07:45:00', 'Low'] = 1.0990
    # UK Session Spike
    df.loc[f'{pattern_day_short} 08:00:00', 'Open'] = 1.1010
    df.loc[f'{pattern_day_short} 08:00:00', 'High'] = 1.1025 # Spike High
    df.loc[f'{pattern_day_short} 08:00:00', 'Low'] = 1.0995
    df.loc[f'{pattern_day_short} 08:00:00', 'Close'] = 1.1005 # Previous candle close
    # Bearish Engulfing
    df.loc[f'{pattern_day_short} 08:15:00', 'Open'] = 1.1020 # Open above previous close
    df.loc[f'{pattern_day_short} 08:15:00', 'High'] = 1.1022
    df.loc[f'{pattern_day_short} 08:15:00', 'Low'] = 1.0985
    df.loc[f'{pattern_day_short} 08:15:00', 'Close'] = 1.0988 # Close below previous open and Asia High

    # --- Inject a textbook long pattern ---
    pattern_day_long = '2023-01-12'
    # Asia Range
    df.loc[f'{pattern_day_long} 00:00:00':f'{pattern_day_long} 07:45:00', ['Open', 'Close']] = 1.1200
    df.loc[f'{pattern_day_long} 00:00:00':f'{pattern_day_long} 07:45:00', 'High'] = 1.1210
    df.loc[f'{pattern_day_long} 00:00:00':f'{pattern_day_long} 07:45:00', 'Low'] = 1.1190
    # UK Session Spike
    df.loc[f'{pattern_day_long} 08:00:00', 'Open'] = 1.1190
    df.loc[f'{pattern_day_long} 08:00:00', 'Low'] = 1.1175 # Spike Low
    df.loc[f'{pattern_day_long} 08:00:00', 'High'] = 1.1205
    df.loc[f'{pattern_day_long} 08:00:00', 'Close'] = 1.1195
    # Bullish Engulfing
    df.loc[f'{pattern_day_long} 08:15:00', 'Open'] = 1.1180 # Open below previous close
    df.loc[f'{pattern_day_long} 08:15:00', 'Low'] = 1.1178
    df.loc[f'{pattern_day_long} 08:15:00', 'High'] = 1.1215
    df.loc[f'{pattern_day_long} 08:15:00', 'Close'] = 1.1212 # Close above previous open and Asia Low

    return df

# --- Preprocessing ---
def preprocess_data(df):
    """Calculates session ranges and other necessary indicators."""
    df['hour'] = df.index.hour
    df['day'] = df.index.date

    # Identify sessions
    df['is_asia'] = (df['hour'] >= ASIA_START_HOUR) & (df['hour'] < ASIA_END_HOUR)
    df['is_uk'] = (df['hour'] >= UK_START_HOUR) & (df['hour'] < UK_END_HOUR)

    # Calculate daily Asia high and low
    asia_session_data = df[df['is_asia']].groupby('day').agg(
        asia_high=('High', 'max'),
        asia_low=('Low', 'min')
    )

    # Calculate Asia range size
    asia_session_data['asia_range'] = (asia_session_data['asia_high'] - asia_session_data['asia_low']) / asia_session_data['asia_low']

    # Merge back into main dataframe
    df = df.merge(asia_session_data, left_on='day', right_index=True, how='left')

    # Forward fill the values for the whole day
    df[['asia_high', 'asia_low', 'asia_range']] = df[['asia_high', 'asia_low', 'asia_range']].ffill()

    # Drop rows with NaN asia_high (first day)
    df.dropna(subset=['asia_high'], inplace=True)

    # Cleanup
    df.drop(columns=['hour', 'day', 'is_asia'], inplace=True)

    return df

# --- Strategy Definition ---
class UkSessionAsiaLiquidityReversalStrategy(Strategy):
    """
    A strategy that looks for liquidity grabs above/below the Asia session range
    during the UK session, confirmed by a reversal candlestick pattern.
    """
    asia_range_max_percent = 2.0  # Max Asia range in percent
    sl_buffer_pips = 5.0          # Buffer for stop loss in pips

    def init(self):
        # Convert optimizable params from float to what's needed
        self.asia_range_max = self.asia_range_max_percent / 100.0
        self.sl_buffer = self.sl_buffer_pips * 0.0001 # Assuming standard pip size for EURUSD-like pair

        # Pass-through pre-calculated data to the strategy scope
        # This is the idiomatic way to use pre-calculated columns in backtesting.py
        self.is_uk = self.I(lambda x: x, self.data.df['is_uk'].values, name='is_uk')
        self.asia_high = self.I(lambda x: x, self.data.df['asia_high'].values, name='asia_high')
        self.asia_low = self.I(lambda x: x, self.data.df['asia_low'].values, name='asia_low')
        self.asia_range = self.I(lambda x: x, self.data.df['asia_range'].values, name='asia_range')

    def next(self):
        # --- Aliases for previous and current candle ---
        prev_open = self.data.Open[-2]
        prev_close = self.data.Close[-2]
        prev_high = self.data.High[-2]
        prev_low = self.data.Low[-2]

        current_open = self.data.Open[-1]
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]

        # --- General Conditions ---
        # Must be in UK session, not in a position, and Asia range must be within limits
        if not self.is_uk[-1] or self.position or self.asia_range[-1] > self.asia_range_max:
            return

        # --- SHORT ENTRY LOGIC ---
        # 1. Price spiked above Asia High on the previous candle
        liquidity_grab_up = prev_high > self.asia_high[-1]

        # 2. A bearish engulfing pattern formed on the current candle
        #    - Current is a down candle
        #    - Current open is > previous close
        #    - Current close is < previous open
        is_bearish_engulfing = (current_close < current_open and
                                current_open >= prev_close and
                                current_close < prev_open)

        # 3. Confirmation: Current candle closed back below Asia High
        confirmation_short = current_close < self.asia_high[-1]

        if liquidity_grab_up and is_bearish_engulfing and confirmation_short:
            sl = current_high + self.sl_buffer
            tp = self.asia_low[-1]

            # Risk/Reward check (optional but good practice)
            if abs(current_close - sl) > 0 and abs(tp - current_close) / abs(current_close - sl) >= 1.0:
                 self.sell(sl=sl, tp=tp)

        # --- LONG ENTRY LOGIC ---
        # 1. Price spiked below Asia Low on the previous candle
        liquidity_grab_down = prev_low < self.asia_low[-1]

        # 2. A bullish engulfing pattern formed on the current candle
        #    - Current is an up candle
        #    - Current open is < previous close
        #    - Current close is > previous open
        is_bullish_engulfing = (current_close > current_open and
                                current_open <= prev_close and
                                current_close > prev_open)

        # 3. Confirmation: Current candle closed back above Asia Low
        confirmation_long = current_close > self.asia_low[-1]

        if liquidity_grab_down and is_bullish_engulfing and confirmation_long:
            sl = current_low - self.sl_buffer
            tp = self.asia_high[-1]

            # Risk/Reward check
            if abs(current_close - sl) > 0 and abs(tp - current_close) / abs(current_close - sl) >= 1.0:
                self.buy(sl=sl, tp=tp)

# --- Backtesting Execution ---
if __name__ == '__main__':
    # --- 1. Data Loading and Preprocessing ---
    # Using synthetic data for a reliable test
    data = generate_synthetic_data()
    data = preprocess_data(data)

    # --- 2. Backtest Initialization ---
    bt = Backtest(data, UkSessionAsiaLiquidityReversalStrategy, cash=100_000, commission=.002)

    # --- 3. Optimization ---
    print("Optimizing strategy...")
    stats = bt.optimize(
        asia_range_max_percent=np.arange(0.5, 3.0, 0.5).tolist(),
        sl_buffer_pips=np.arange(2, 11, 2).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.asia_range_max_percent > 0 and p.sl_buffer_pips > 0
    )
    print("Best stats found:")
    print(stats)

    # --- 4. JSON Output Handling ---
    # A robust function to sanitize results for JSON serialization
    def sanitize_for_json(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
             return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
             return obj.to_dict(orient='records')
        elif pd.isna(obj):
            return None
        return obj

    # Sanitize the stats dictionary/series
    sanitized_stats = {key: sanitize_for_json(value) for key, value in stats.items()}

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Save the sanitized results
    output_path = 'results/temp_result.json'
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump({
            'strategy_name': 'uk_session_asia_liquidity_reversal',
            'return': sanitized_stats.get('Return [%]', None),
            'sharpe': sanitized_stats.get('Sharpe Ratio', None),
            'max_drawdown': sanitized_stats.get('Max. Drawdown [%]', None),
            'win_rate': sanitized_stats.get('Win Rate [%]', None),
            'total_trades': sanitized_stats.get('# Trades', 0)
        }, f, indent=4)

    # --- 5. Plotting ---
    plot_path = 'results/uk_session_asia_liquidity_reversal.html'
    print(f"Generating plot... saved to {plot_path}")
    try:
        bt.plot(filename=plot_path, open_browser=False)
    except TypeError as e:
        print(f"Could not generate plot due to a known issue with backtesting.py and pandas: {e}")
        print("Continuing without plot...")

    print("Backtest complete.")
