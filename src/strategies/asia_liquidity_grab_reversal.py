
import json
import os

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

# Optimal pandas display options for debugging
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame to add indicators and session data
    needed for the Asia Liquidity Grab Reversal Strategy.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Data index must be a DatetimeIndex.")

    # --- Session Identification ---
    df['hour'] = df.index.hour
    df['day'] = df.index.date

    df['session'] = 'Other'
    df.loc[df['hour'].between(0, 7), 'session'] = 'Asia'
    df.loc[df['hour'].between(8, 15), 'session'] = 'UK'
    # US Session (13:00-21:00 UTC) can overlap with UK
    df.loc[df['hour'].between(13, 21), 'session'] = 'US'

    df['trade_session'] = df['session'].isin(['UK', 'US'])

    # --- Asia Session Range Calculation ---
    # Group by day and get the high/low of the 'Asia' session for that day
    asia_df = df[df['session'] == 'Asia']
    asia_session_data = asia_df.groupby(asia_df.index.date)
    daily_asia_high = asia_session_data['High'].max()
    daily_asia_low = asia_session_data['Low'].min()

    # Map the daily Asia high/low to each row for the entire day
    df['asia_high'] = pd.Series(df.index.date, index=df.index).map(daily_asia_high)
    df['asia_low'] = pd.Series(df.index.date, index=df.index).map(daily_asia_low)

    # Forward-fill the values to apply yesterday's Asia range to today's sessions
    df[['asia_high', 'asia_low']] = df[['asia_high', 'asia_low']].fillna(method='ffill')

    # Calculate Asia range as a percentage
    df['asia_range_percent'] = (df['asia_high'] - df['asia_low']) / df['asia_low'] * 100

    # --- Candlestick Pattern Identification ---
    body_size = abs(df['Close'] - df['Open'])
    candle_range = df['High'] - df['Low']

    # Bearish Engulfing
    is_bearish_engulfing = (df['Open'].shift(1) < df['Close'].shift(1)) & \
                             (df['Open'] > df['Close']) & \
                             (df['Open'] > df['Close'].shift(1)) & \
                             (df['Close'] < df['Open'].shift(1))

    # Bullish Engulfing
    is_bullish_engulfing = (df['Open'].shift(1) > df['Close'].shift(1)) & \
                             (df['Open'] < df['Close']) & \
                             (df['Open'] < df['Close'].shift(1)) & \
                             (df['Close'] > df['Open'].shift(1))

    # Shooting Star (a type of bearish pin bar)
    upper_wick = df['High'] - pd.concat([df['Open'], df['Close']], axis=1).max(axis=1)
    is_shooting_star = (upper_wick > 2 * body_size) & (body_size > 0) & (candle_range > 0)

    # Hammer (a type of bullish pin bar)
    lower_wick = pd.concat([df['Open'], df['Close']], axis=1).min(axis=1) - df['Low']
    is_hammer = (lower_wick > 2 * body_size) & (body_size > 0) & (candle_range > 0)

    df['is_bearish_reversal'] = is_bearish_engulfing | is_shooting_star
    df['is_bullish_reversal'] = is_bullish_engulfing | is_hammer

    # --- Daily/Weekly Levels (for future TP targets, not used in TP1) ---
    # Note: These are not used for the primary TP1 logic but included as per requirements.
    # We use .shift(1) to prevent lookahead bias.
    daily_resampler = df['Close'].resample('D')
    df['prev_daily_high'] = daily_resampler.max().shift(1).reindex(df.index, method='ffill')
    df['prev_daily_low'] = daily_resampler.min().shift(1).reindex(df.index, method='ffill')
    df['daily_50_level'] = (df['prev_daily_high'] + df['prev_daily_low']) / 2

    weekly_resampler = df['Close'].resample('W')
    df['prev_weekly_high'] = weekly_resampler.max().shift(1).reindex(df.index, method='ffill')
    df['prev_weekly_low'] = weekly_resampler.min().shift(1).reindex(df.index, method='ffill')
    df['weekly_50_level'] = (df['prev_weekly_high'] + df['prev_weekly_low']) / 2

    df.dropna(inplace=True)
    return df

class AsiaLiquidityGrabReversalStrategy(Strategy):
    """
    A trading strategy that looks for liquidity grabs above/below the
    Asia session range high/low, followed by a reversal pattern during
    the UK or US sessions.
    """
    # Optimization parameters
    asia_range_max_percent = 2.0  # Max Asia range in percent

    def init(self):
        # The pre-processing function has already prepared the data,
        # so we can access the columns directly from self.data.df.
        # Using self.I() is not necessary for pre-calculated columns.
        pass

    def next(self):
        # Ensure we have enough data and not in an open position
        if self.position:
            return

        # Get the most recent data point
        # Use .iloc[-1] to access pandas Series values from the pre-processed df
        current_data = self.data.df.iloc[-1]

        # --- Strategy Filters ---
        # 1. Trade only during UK or US sessions
        if not current_data['trade_session']:
            return

        # 2. Asia session range must be less than the max percentage
        if current_data['asia_range_percent'] >= self.asia_range_max_percent:
            return

        # --- Entry Logic ---

        # Short Entry: Price spikes above Asia High, then a bearish reversal candle
        if (self.data.High[-1] > current_data['asia_high'] and
            self.data.Close[-1] < current_data['asia_high'] and
            current_data['is_bearish_reversal']):

            sl = self.data.High[-1] * 1.001 # Stop loss just above the spike high
            tp = current_data['asia_low']   # Take profit at the other side of the range

            # Validate TP/SL before placing trade
            if self.data.Close[-1] > tp and self.data.Close[-1] < sl:
                self.sell(sl=sl, tp=tp)

        # Long Entry: Price spikes below Asia Low, then a bullish reversal candle
        elif (self.data.Low[-1] < current_data['asia_low'] and
              self.data.Close[-1] > current_data['asia_low'] and
              current_data['is_bullish_reversal']):

            sl = self.data.Low[-1] * 0.999 # Stop loss just below the spike low
            tp = current_data['asia_high'] # Take profit at the other side of the range

            # Validate TP/SL before placing trade
            if self.data.Close[-1] < tp and self.data.Close[-1] > sl:
                self.buy(sl=sl, tp=tp)


def generate_synthetic_forex_data(days=100, freq='15min'):
    """Generates synthetic 24-hour data that mimics a Forex pair."""
    rng = np.random.default_rng(seed=42)
    n_points = days * 24 * (60 // int(freq.replace('min', '')))

    # Create a DatetimeIndex
    index = pd.to_datetime('2023-01-01') + pd.to_timedelta(
        np.arange(n_points), unit='m') * int(freq.replace('min', ''))

    # Generate price movements with some randomness and trend
    returns = rng.normal(loc=0.0001, scale=0.002, size=n_points)
    price = 1.1000 * (1 + returns).cumprod()

    # Create OHLC data
    df = pd.DataFrame(index=index)
    df['Close'] = price
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])

    # Generate High and Low based on Close and Open
    df['High'] = pd.concat([df['Open'], df['Close']], axis=1).max(axis=1) + \
                 rng.uniform(0.0001, 0.001, size=n_points)
    df['Low'] = pd.concat([df['Open'], df['Close']], axis=1).min(axis=1) - \
                rng.uniform(0.0001, 0.001, size=n_points)

    return df

if __name__ == '__main__':
    # 1. Load or generate data
    # Using synthetic data because GOOG data has gaps and is not 24h
    data = generate_synthetic_forex_data(days=180, freq='15min')

    # 2. Preprocess the data
    processed_data = preprocess_data(data.copy())

    # 3. Run backtest
    bt = Backtest(processed_data, AsiaLiquidityGrabReversalStrategy, cash=100000, commission=.0002)

    # 4. Optimize
    # Note: Optimization can be slow. A smaller range or smaller dataset is faster.
    stats = bt.optimize(
        asia_range_max_percent=np.arange(1.0, 3.5, 0.5).tolist(),
        maximize='Sharpe Ratio'
    )

    print("--- Best Stats ---")
    print(stats)
    print("--- Best Strategy Parameters ---")
    print(stats._strategy)

    # 5. Save results
    os.makedirs('results', exist_ok=True)

    # Handle cases where no trades are made
    if stats['# Trades'] > 0:
        result_dict = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }
    else:
        result_dict = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0
        }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

    print("\nResults saved to results/temp_result.json")

    # 6. Generate plot
    bt.plot(filename='results/asia_liquidity_grab_reversal.html', open_browser=False)
    print("Plot saved to results/asia_liquidity_grab_reversal.html")
