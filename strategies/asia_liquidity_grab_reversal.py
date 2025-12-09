import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from enum import Enum


# --- Synthetic Data Generation ---
def generate_synthetic_data(days=100):
    """
    Generates synthetic 15-minute data with clear Asia and UK session dynamics
    to test the Asia Liquidity Grab Reversal strategy.
    """
    rng = np.random.default_rng(42)
    start_date = '2023-01-01'
    total_bars = days * 24 * 4  # 96 bars per day (15-min timeframe)

    # Create a DatetimeIndex
    index = pd.to_datetime(pd.date_range(start=start_date, periods=total_bars, freq='15min', tz='UTC'))

    # Generate base price movement
    price = 100 + np.cumsum(rng.normal(0, 0.02, total_bars))

    # Create DataFrame
    df = pd.DataFrame(index=index)
    df['Open'] = price
    df['High'] = price
    df['Low'] = price
    df['Close'] = price

    # Define session times (UTC)
    asia_start, asia_end = 2, 8  # Simplified: 02:00 - 08:00
    uk_start, uk_end = 8, 12     # Simplified: 08:00 - 12:00 for grab event

    # Inject session-specific patterns
    for i in range(1, len(df)):
        hour = df.index[i].hour

        # Asia Session: Low volatility range
        if asia_start <= hour < asia_end:
            df.iloc[i] = df.iloc[i-1]['Close'] + rng.normal(0, 0.01)

        # UK Session Start: Liquidity grab event
        elif uk_start <= hour < uk_end and uk_start <= df.index[i-1].hour < uk_end:
             # Check if the previous day's Asia session is defined
            today = df.index[i].date()
            asia_session_mask = (df.index.date == today) & (df.index.hour >= asia_start) & (df.index.hour < asia_end)

            if asia_session_mask.any():
                asia_high = df[asia_session_mask]['High'].max()
                asia_low = df[asia_session_mask]['Low'].min()

                # 50% chance of a grab event
                if rng.random() > 0.5:
                    is_high_grab = rng.random() > 0.5

                    if is_high_grab and not np.isnan(asia_high):
                        # Spike above Asia High
                        spike = asia_high + rng.uniform(0.02, 0.05)
                        reversal_close = asia_high - rng.uniform(0.01, 0.03)
                        df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i-1]['Close'], spike, reversal_close)
                        df.iloc[i, df.columns.get_loc('Open')] = df.iloc[i-1]['Close']
                        df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i-1]['Close'], spike, reversal_close)
                        df.iloc[i, df.columns.get_loc('Close')] = reversal_close
                    elif not is_high_grab and not np.isnan(asia_low):
                        # Spike below Asia Low
                        spike = asia_low - rng.uniform(0.02, 0.05)
                        reversal_close = asia_low + rng.uniform(0.01, 0.03)
                        df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i-1]['Close'], spike, reversal_close)
                        df.iloc[i, df.columns.get_loc('Open')] = df.iloc[i-1]['Close']
                        df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i-1]['Close'], spike, reversal_close)
                        df.iloc[i, df.columns.get_loc('Close')] = reversal_close
        else:
            # Default random walk
            df.iloc[i] = df.iloc[i-1]['Close'] + rng.normal(0, 0.02, 4)

    # Refine OHLC
    for i in range(len(df)):
        vals = df.iloc[i, [df.columns.get_loc('Open'), df.columns.get_loc('High'), df.columns.get_loc('Low'), df.columns.get_loc('Close')]].values
        rng.shuffle(vals)
        o, c = vals[0], vals[1]
        df.iloc[i, df.columns.get_loc('Open')] = o
        df.iloc[i, df.columns.get_loc('Close')] = c
        df.iloc[i, df.columns.get_loc('High')] = max(o, c) + rng.uniform(0, 0.01)
        df.iloc[i, df.columns.get_loc('Low')] = min(o, c) - rng.uniform(0, 0.01)

    df = df.astype(float)
    # Ensure Volume column exists as it's required by backtesting.py
    if 'Volume' not in df.columns:
        df['Volume'] = rng.integers(100, 1000, size=len(df))

    return df.dropna()

# --- Data Pre-processing ---
def preprocess_data(df: pd.DataFrame):
    """
    Calculates and maps the previous Asia session's high, low, and range
    to each bar of the day. Uses integer codes for dates to avoid
    multiprocessing serialization issues.
    """
    # Create integer codes for each date to use as a grouper
    # This is more robust for multiprocessing in bt.optimize()
    date_codes, dates = pd.factorize(df.index.date)
    df['date_codes'] = date_codes

    # Define session times (UTC)
    asia_start_hour, asia_end_hour = 0, 8

    # Calculate daily Asia session stats using the integer codes
    asia_session_mask = (df.index.hour >= asia_start_hour) & (df.index.hour < asia_end_hour)

    # Group the filtered data by its own date_codes
    asia_stats = df[asia_session_mask].groupby('date_codes').agg(
        asia_high=pd.NamedAgg(column='High', aggfunc='max'),
        asia_low=pd.NamedAgg(column='Low', aggfunc='min')
    )

    # Calculate Asia range and percentage
    asia_stats['asia_range'] = asia_stats['asia_high'] - asia_stats['asia_low']
    asia_stats['asia_range_perc'] = (asia_stats['asia_range'] / asia_stats['asia_low']) * 100

    # Map the stats back to the main dataframe using the date codes
    # NO SHIFT: The strategy uses the current day's Asia range.
    df['asia_high'] = df['date_codes'].map(asia_stats['asia_high'])
    df['asia_low'] = df['date_codes'].map(asia_stats['asia_low'])
    df['asia_range'] = df['date_codes'].map(asia_stats['asia_range'])
    df['asia_range_perc'] = df['date_codes'].map(asia_stats['asia_range_perc'])

    # Forward-fill the daily stats across all bars of the day
    df[['asia_high', 'asia_low', 'asia_range', 'asia_range_perc']] = df[['asia_high', 'asia_low', 'asia_range', 'asia_range_perc']].ffill()

    return df.drop(columns=['date_codes'])

# --- Helper Functions ---
def is_bearish_engulfing(data):
    """Checks if the last candle is a bearish engulfing pattern."""
    if len(data.Open) < 2:
        return False

    current_open = data.Open[-1]
    current_close = data.Close[-1]
    prev_open = data.Open[-2]
    prev_close = data.Close[-2]

    # Current is a bearish candle
    if current_close >= current_open:
        return False

    # Previous is a bullish candle
    if prev_close <= prev_open:
        return False

    # Current bearish body engulfs previous bullish body
    return current_open > prev_close and current_close < prev_open

def is_bullish_engulfing(data):
    """Checks if the last candle is a bullish engulfing pattern."""
    if len(data.Open) < 2:
        return False

    current_open = data.Open[-1]
    current_close = data.Close[-1]
    prev_open = data.Open[-2]
    prev_close = data.Close[-2]

    # Current is a bullish candle
    if current_close <= current_open:
        return False

    # Previous is a bearish candle
    if prev_close >= prev_open:
        return False

    # Current bullish body engulfs previous bearish body
    return current_open < prev_close and current_close > prev_open

# --- Strategy Class ---
class AsiaLiquidityGrabReversalStrategy(Strategy):
    """
    A strategy that trades reversals after a liquidity grab above/below the
    Asia session high/low during the UK session.
    """
    # Optimization parameters
    max_asia_range_perc = 2.0
    sl_buffer_pip = 1.0 # Stop loss buffer in pips (0.01)
    tp1_portion = 0.5   # Portion of position to close at TP1

    # --- Strategy state variables ---
    tp1_price = None
    tp1_hit = False

    def init(self):
        # Map the pre-processed data columns to indicators for easy access.
        self.asia_high = self.I(lambda x: x, self.data.df['asia_high'].values, name="asia_high")
        self.asia_low = self.I(lambda x: x, self.data.df['asia_low'].values, name="asia_low")
        self.asia_range_perc = self.I(lambda x: x, self.data.df['asia_range_perc'].values, name="asia_range_perc")

        self.uk_session_start_hour = 8
        self.uk_session_end_hour = 12

    def next(self):
        # Get data for the current bar
        current_hour = self.data.index[-1].hour
        asia_high = self.asia_high[-1]
        asia_low = self.asia_low[-1]
        asia_range_perc = self.asia_range_perc[-1]

        # --- Position Management ---
        if self.position:
            # Check if TP1 has been hit
            if not self.tp1_hit:
                if self.position.is_long and self.data.High[-1] >= self.tp1_price:
                    self.position.close(portion=self.tp1_portion)
                    self.trades[0].sl = self.trades[0].entry_price # Move SL to BE
                    self.tp1_hit = True
                elif self.position.is_short and self.data.Low[-1] <= self.tp1_price:
                    self.position.close(portion=self.tp1_portion)
                    self.trades[0].sl = self.trades[0].entry_price
                    self.tp1_hit = True
            # Allow position to run until SL is hit or it's manually closed
            return

        # --- Entry Filters ---
        is_uk_session = self.uk_session_start_hour <= current_hour < self.uk_session_end_hour
        is_range_ok = not pd.isna(asia_range_perc) and asia_range_perc < self.max_asia_range_perc
        if not is_uk_session or not is_range_ok:
            return

        # --- Entry Logic ---
        # SHORT ENTRY
        is_grab_high = self.data.High[-1] > asia_high and self.data.Close[-1] < asia_high
        if is_grab_high and is_bearish_engulfing(self.data):
            sl_price = self.data.High[-1] + (self.sl_buffer_pip * 0.01)
            self.tp1_price = asia_low

            if sl_price > self.data.Close[-1] and self.tp1_price < self.data.Close[-1]:
                self.sell(sl=sl_price)
                self.tp1_hit = False # Reset TP flag on new trade
            return

        # LONG ENTRY
        is_grab_low = self.data.Low[-1] < asia_low and self.data.Close[-1] > asia_low
        if is_grab_low and is_bullish_engulfing(self.data):
            sl_price = self.data.Low[-1] - (self.sl_buffer_pip * 0.01)
            self.tp1_price = asia_high

            if sl_price < self.data.Close[-1] and self.tp1_price > self.data.Close[-1]:
                self.buy(sl=sl_price)
                self.tp1_hit = False # Reset TP flag on new trade
            return


if __name__ == '__main__':
    # 1. Generate and Preprocess Data
    data = generate_synthetic_data(days=200)
    data = preprocess_data(data)

    # 2. Run Backtest
    # Use a larger cash amount to avoid margin errors with default sizing
    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=100_000, commission=.002)

    # 3. Optimize
    # Note: Use a list/tuple for optimizable parameters
    stats = bt.optimize(
        max_asia_range_perc=[1.0, 1.5, 2.0, 2.5],
        sl_buffer_pip=[1, 2, 3],
        tp1_portion=[0.25, 0.5, 0.75],
        maximize='Sharpe Ratio'
    )

    print("Best run stats:")
    print(stats)

    # 4. Save Results
    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize results for JSON output
    results_dict = {
        'strategy_name': 'asia_liquidity_grab_reversal',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    for key, value in results_dict.items():
        if pd.isna(value):
            results_dict[key] = None # Replace NaN with null for JSON
        elif isinstance(value, (np.int64, np.int32)):
            results_dict[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            results_dict[key] = float(value)

    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("\nResults saved to results/temp_result.json")

    # 5. Generate Plot
    try:
        bt.plot(filename="results/asia_liquidity_grab_reversal.html")
        print("Plot saved to results/asia_liquidity_grab_reversal.html")
    except TypeError as e:
        print(f"\nCould not generate plot due to a known issue with backtesting.py and pandas: {e}")
        print("However, the JSON results have been saved successfully.")
