
import json
import os
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

class AsiaLiquidityGrabUkSessionStrategy(Strategy):
    """
    Strategy to trade the Asia session liquidity grab during the UK session.
    """

    # Strategy parameters for optimization
    asia_range_max_perc = 2.0 # Max Asia session range in percent

    def init(self):
        """
        Initialize the strategy's state variables and indicators.
        """
        # Map pre-calculated data columns to indicators to avoid look-ahead bias
        self.is_uk = self.I(identity, self.data.df['is_uk'].values)
        self.asia_high = self.I(identity, self.data.df['asia_high'].values)
        self.asia_low = self.I(identity, self.data.df['asia_low'].values)
        self.asia_range_perc = self.I(identity, self.data.df['asia_range_perc'].values)

        # State variables
        self.liquidity_grab_high = None
        self.current_day = None

    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        # Get the current date
        today = self.data.index[-1].date()

        # Reset state at the start of a new day
        if self.current_day != today:
            self.current_day = today
            self.liquidity_grab_high = None

        # --- Conditions for Entry ---

        # 1. Must be in UK session.
        if not self.is_uk[-1]:
            return

        # 2. No open positions
        if self.position:
            return

        # 3. Asia range must be within the defined limit
        if self.asia_range_perc[-1] > self.asia_range_max_perc:
            return

        # --- State Machine for Entry ---

        # State 1: Detect the liquidity grab
        # Price breaks above the Asia session high
        if self.data.High[-1] > self.asia_high[-1]:
            if self.liquidity_grab_high is None:
                 self.liquidity_grab_high = self.data.High[-1]
            else:
                 self.liquidity_grab_high = max(self.liquidity_grab_high, self.data.High[-1])

        # State 2: Look for a bearish engulfing pattern after the grab
        if self.liquidity_grab_high is not None:
            # Bearish Engulfing Pattern Check:
            if len(self.data.Close) < 2:
                return

            current_open = self.data.Open[-1]
            current_close = self.data.Close[-1]
            previous_open = self.data.Open[-2]
            previous_close = self.data.Close[-2]

            is_bearish_candle = current_close < current_open
            engulfs_previous = current_open >= previous_close and current_close < previous_open

            if is_bearish_candle and engulfs_previous:

                # --- Place Trade ---

                stop_loss = self.liquidity_grab_high * 1.0005 # Add a small buffer
                take_profit = self.asia_low[-1]
                entry_price = self.data.Close[-1]

                # Risk Management Check: TP must be below entry for a short
                if take_profit < entry_price:
                    self.sell(sl=stop_loss, tp=take_profit)

                # Reset state after placing trade
                self.liquidity_grab_high = None

def identity(series):
    """Helper function to pass pre-calculated series to self.I"""
    return series

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds session information and calculates Asia session high/low.
    Assumes the DataFrame index is a DatetimeIndex in UTC.
    """
    df = df.copy()

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe index must be a DatetimeIndex.")

    if df.index.tz:
        df.index = df.index.tz_localize(None)

    # Define session times in UTC
    asia_start = pd.to_datetime('00:00').time()
    asia_end = pd.to_datetime('08:00').time()
    uk_start = pd.to_datetime('07:00').time()
    uk_end = pd.to_datetime('16:00').time()

    # Identify sessions
    df['is_asia'] = (df.index.time >= asia_start) & (df.index.time < asia_end)
    df['is_uk'] = (df.index.time >= uk_start) & (df.index.time < uk_end)

    # Calculate daily Asia session stats using an index-preserving method
    asia_session_df = df[df['is_asia']]
    daily_asia_stats = asia_session_df.groupby(asia_session_df.index.date).agg(
        asia_high=('High', 'max'),
        asia_low=('Low', 'min')
    )

    # Map stats back to the main DataFrame
    df['asia_high'] = df.index.to_series().dt.date.map(daily_asia_stats['asia_high'])
    df['asia_low'] = df.index.to_series().dt.date.map(daily_asia_stats['asia_low'])

    df['asia_range_perc'] = (
        (df['asia_high'] - df['asia_low']) / df['asia_low']
    ) * 100

    # Forward-fill the daily values to apply them to the whole day
    df[['asia_high', 'asia_low', 'asia_range_perc']] = df[['asia_high', 'asia_low', 'asia_range_perc']].ffill()

    # Drop rows with NaN values that couldn't be filled (e.g., first day)
    df.dropna(inplace=True)

    return df


def generate_synthetic_data(days=30):
    """
    Generates synthetic 15-minute OHLC data tailored to test the strategy.
    This function will create a textbook "Asia Liquidity Grab" pattern.
    """
    n_periods = days * 24 * 4  # 30 days, 24 hours, 4 periods per hour
    rng = pd.date_range('2023-01-01', periods=n_periods, freq='15min')
    df = pd.DataFrame(index=rng)

    # Base price and volatility
    base_price = 1.1000
    volatility = 0.0005

    # Generate random walk for price
    random_walk = np.random.normal(loc=0, scale=volatility, size=n_periods).cumsum()
    df['Close'] = base_price + random_walk

    # --- Create the specific pattern for a few days ---

    # Select a few days to inject the pattern
    pattern_days = pd.to_datetime(['2023-01-05', '2023-01-15', '2023-01-25'])

    for day in pattern_days:

        # --- 1. Asia Session: Create a tight range ---
        asia_start_time = day.replace(hour=0, minute=0)
        asia_end_time = day.replace(hour=8, minute=0)
        asia_mask = (df.index >= asia_start_time) & (df.index < asia_end_time)

        # Define the range
        asia_high = base_price + 0.0020
        asia_low = base_price - 0.0010

        # Force price into this range during Asia session
        df.loc[asia_mask, 'Close'] = np.linspace(asia_low, asia_high, asia_mask.sum())

        # --- 2. UK Session: Create the liquidity grab and reversal ---
        uk_start_time = day.replace(hour=8, minute=15)

        # Grab candle (breaks above Asia high)
        grab_time = uk_start_time
        grab_idx = df.index.get_loc(grab_time)
        df.loc[grab_time, 'Close'] = asia_high + 0.0005

        # Engulfing candle (strong reversal)
        engulfing_time = uk_start_time + pd.Timedelta(minutes=15)
        engulfing_idx = df.index.get_loc(engulfing_time)
        df.loc[engulfing_time, 'Close'] = df.loc[grab_time, 'Close'] - 0.0015

        # --- 3. Subsequent candles move towards Asia low ---
        target_time_start = uk_start_time + pd.Timedelta(minutes=30)
        target_time_end = day.replace(hour=12, minute=0)
        target_mask = (df.index >= target_time_start) & (df.index < target_time_end)

        num_target_candles = target_mask.sum()
        if num_target_candles > 0:
            df.loc[target_mask, 'Close'] = np.linspace(
                df.loc[engulfing_time, 'Close'],
                asia_low,
                num_target_candles
            )

    # --- Generate OHLC from Close price ---
    df['Open'] = df['Close'].shift(1)
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, volatility, len(df))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, volatility, len(df))
    df['Volume'] = np.random.randint(100, 1000, size=len(df))

    # Fill the first row's NaN
    df.iloc[0, df.columns.get_loc('Open')] = df.iloc[0, df.columns.get_loc('Close')]
    df.dropna(inplace=True)

    return df

if __name__ == '__main__':

    data = generate_synthetic_data()
    data = preprocess_data(data)

    # Initialize and run the backtest
    bt = Backtest(data, AsiaLiquidityGrabUkSessionStrategy, cash=100_000, commission=.002)

    # Optimize the strategy parameters
    stats = bt.optimize(
        asia_range_max_perc=[i / 10 for i in range(1, 6)], # 0.1 to 0.5
        maximize='Sharpe Ratio',
        constraint=lambda p: p.asia_range_max_perc > 0
    )

    print("Best stats:", stats)

    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Save the results to a JSON file
    # Ensure all values are JSON serializable
    results = {
        'strategy_name': 'asia_liquidity_grab_uk_session',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    for key, value in results.items():
        if isinstance(value, (np.generic, np.ndarray)):
            results[key] = value.item() if value.size == 1 else value.tolist()

    with open('results/temp_result.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate and save the plot
    try:
        bt.plot(filename='results/asia_liquidity_grab_uk_session.html', open_browser=False)
    except TypeError as e:
        print(f"Could not generate plot due to a known issue with backtesting.py and pandas: {e}")

    print("Backtest complete. Results saved to results/temp_result.json and plot to results/asia_liquidity_grab_uk_session.html")
