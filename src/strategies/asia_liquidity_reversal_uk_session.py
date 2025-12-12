import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

# --- Data Preprocessing ---

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds necessary indicators and session data to the DataFrame.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.tz_localize(None).tz_localize('UTC', ambiguous='infer')

    # Session times (UTC)
    ASIA_START_HOUR = 0
    ASIA_END_HOUR = 8
    UK_START_HOUR = 8
    UK_END_HOUR = 16

    # Time-based features
    df['hour'] = df.index.hour
    df['day'] = df.index.date
    df['week'] = df.index.to_period('W').start_time.date

    # In Session flags
    df['is_asia_session'] = (df['hour'] >= ASIA_START_HOUR) & (df['hour'] < ASIA_END_HOUR)
    df['is_uk_session'] = (df['hour'] >= UK_START_HOUR) & (df['hour'] < UK_END_HOUR)

    # Calculate previous day's data
    daily_data = df.groupby('day').agg({
        'High': 'max',
        'Low': 'min'
    }).shift(1) # Use previous day's data
    daily_data.rename(columns={'High': 'prev_day_high', 'Low': 'prev_day_low'}, inplace=True)
    df = df.merge(daily_data, left_on='day', right_index=True, how='left')
    df['prev_day_50'] = (df['prev_day_high'] + df['prev_day_low']) / 2

    # Calculate previous week's data
    weekly_data = df.groupby(pd.Grouper(freq='W')).agg({
        'High': 'max',
        'Low': 'min'
    }).shift(1) # Use previous week's data
    weekly_data.rename(columns={'High': 'prev_week_high', 'Low': 'prev_week_low'}, inplace=True)
    # Create a weekly date column to merge on
    df['week_start'] = df.index.to_period('W').start_time
    weekly_data.index = weekly_data.index.to_period('W').start_time
    df = df.merge(weekly_data, left_on='week_start', right_index=True, how='left')
    df['prev_week_50'] = (df['prev_week_high'] + df['prev_week_low']) / 2

    # Identify Asia session High and Low
    asia_session_stats = df[df['is_asia_session']].groupby('day').agg({
        'High': 'max',
        'Low': 'min'
    })
    asia_session_stats.rename(columns={'High': 'asia_high', 'Low': 'asia_low'}, inplace=True)

    df = df.merge(asia_session_stats, left_on='day', right_index=True, how='left')

    # Forward fill the values for the entire day
    df['asia_high'] = df.groupby('day')['asia_high'].ffill()
    df['asia_low'] = df.groupby('day')['asia_low'].ffill()

    # Asia Range Calculation
    df['asia_range'] = ((df['asia_high'] - df['asia_low']) / df['asia_low']) * 100

    # Candlestick Patterns
    # Bearish Engulfing
    df['is_bearish_engulfing'] = (df['Open'] > df['Close']) & \
                               (df['Close'].shift(1) > df['Open'].shift(1)) & \
                               (df['Open'] > df['Close'].shift(1)) & \
                               (df['Close'] < df['Open'].shift(1))

    # Bullish Engulfing
    df['is_bullish_engulfing'] = (df['Close'] > df['Open']) & \
                                (df['Open'].shift(1) > df['Close'].shift(1)) & \
                                (df['Close'] > df['Open'].shift(1)) & \
                                (df['Open'] < df['Close'].shift(1))

    # Drop rows with NaN values created by shifts and merges
    df.dropna(inplace=True)

    return df

# --- Strategy ---

class AsiaLiquidityReversalUkSessionStrategy(Strategy):
    asia_range_max = 2.0
    sl_cushion = 1.001

    def init(self):
        # Alias data columns for easier access
        self.is_uk_session = self.I(lambda x: x, self.data.df.is_uk_session, plot=False)
        self.asia_high = self.I(lambda x: x, self.data.df.asia_high, plot=True)
        self.asia_low = self.I(lambda x: x, self.data.df.asia_low, plot=True)
        self.asia_range = self.I(lambda x: x, self.data.df.asia_range, plot=False)
        self.is_bearish_engulfing = self.I(lambda x: x, self.data.df.is_bearish_engulfing, plot=False)
        self.is_bullish_engulfing = self.I(lambda x: x, self.data.df.is_bullish_engulfing, plot=False)

    def next(self):
        # --- Risk Management ---
        if self.asia_range[-1] >= self.asia_range_max:
            return

        # Only trade during UK session
        if not self.is_uk_session[-1]:
            return

        # Avoid layering positions
        if self.position:
            return

        price = self.data

        # --- SHORT Entry Logic ---
        # 1. Price spiked above Asia High in the current or previous candle
        liquidity_grab_up = price.High[-1] > self.asia_high[-1] or price.High[-2] > self.asia_high[-2]

        # 2. Bearish reversal pattern
        reversal_signal_short = self.is_bearish_engulfing[-1]

        # 3. Reversal candle closes back below Asia High
        closes_below_hoa = price.Close[-1] < self.asia_high[-1]

        if liquidity_grab_up and reversal_signal_short and closes_below_hoa:
            sl = price.High[-1] * self.sl_cushion
            tp = self.asia_low[-1]
            if sl > price.Close[-1] and tp < price.Close[-1]:
                self.sell(sl=sl, tp=tp)

        # --- LONG Entry Logic ---
        # 1. Price spiked below Asia Low in the current or previous candle
        liquidity_grab_down = price.Low[-1] < self.asia_low[-1] or price.Low[-2] < self.asia_low[-2]

        # 2. Bullish reversal pattern
        reversal_signal_long = self.is_bullish_engulfing[-1]

        # 3. Reversal candle closes back above Asia Low
        closes_above_loa = price.Close[-1] > self.asia_low[-1]

        if liquidity_grab_down and reversal_signal_long and closes_above_loa:
            sl = price.Low[-1] / self.sl_cushion
            tp = self.asia_high[-1]
            if sl < price.Close[-1] and tp > price.Close[-1]:
                self.buy(sl=sl, tp=tp)

# --- Backtesting ---

if __name__ == '__main__':
    # Generate synthetic 15-minute data
    np.random.seed(42)
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    date_rng = pd.date_range(start=start_date, end=end_date, freq='15min')

    price_path = np.random.randn(len(date_rng)).cumsum()
    initial_price = 1.5000

    df = pd.DataFrame(index=date_rng)
    df['Open'] = initial_price + price_path + np.random.uniform(-0.001, 0.001, len(date_rng))
    df['High'] = df['Open'] + np.random.uniform(0, 0.005, len(date_rng))
    df['Low'] = df['Open'] - np.random.uniform(0, 0.005, len(date_rng))
    df['Close'] = df['Open'] + np.random.uniform(-0.002, 0.002, len(date_rng))

    # Preprocess the data
    data = preprocess_data(df)

    # Run backtest
    bt = Backtest(data, AsiaLiquidityReversalUkSessionStrategy, cash=100_000, commission=.0002)

    # Optimize
    stats = bt.optimize(
        asia_range_max=np.arange(1.0, 4.0, 0.5).tolist(),
        sl_cushion=np.arange(1.001, 1.005, 0.001).tolist(),
        maximize='Sharpe Ratio'
    )

    print(stats)

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Check if trades were made
    if stats['# Trades'] > 0:
        result_dict = {
            'strategy_name': 'asia_liquidity_reversal_uk_session',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }
    else:
        result_dict = {
            'strategy_name': 'asia_liquidity_reversal_uk_session',
            'return': 0.0,
            'sharpe': None,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0
        }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

    # Generate plot
    bt.plot()