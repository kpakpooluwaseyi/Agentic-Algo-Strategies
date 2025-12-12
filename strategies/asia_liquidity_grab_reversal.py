from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

def generate_forex_data(days=100):
    """
    Generates synthetic 15-minute Forex data with typical session behavior.
    - Asia session: Low volatility, range-bound.
    - London session: Higher volatility, potential for breakouts/reversals.
    - Includes specific liquidity grab patterns for testing.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range(start='2023-01-01', periods=days * 24 * 4, freq='15min')
    df = pd.DataFrame(index=dates)

    # Base price movement
    price = 1.1000
    prices = [price]
    for _ in range(len(dates) - 1):
        price += rng.normal(0, 0.00015)
        prices.append(price)
    df['Close'] = pd.Series(prices, index=dates).rolling(window=10, min_periods=1).mean()

    # Define session-based volatility multipliers
    df['hour'] = df.index.hour
    session_volatility = df['hour'].apply(
        lambda h: 0.5 if 0 <= h < 8 else (1.5 if 8 <= h < 16 else 1.0)
    ).values

    # Inject specific patterns for bearish and bullish reversals
    for day in range(5, days, 20):  # Bearish pattern
        asia_start_str = f'2023-01-01'
        asia_start_time = pd.to_datetime(asia_start_str) + pd.Timedelta(days=day, hours=0)
        asia_end_time = asia_start_time + pd.Timedelta(hours=7, minutes=45)
        london_start_time = asia_end_time + pd.Timedelta(minutes=15)

        base_price = df.loc[str(asia_start_time.date())].iloc[0]['Close']
        asia_high = base_price + 0.0010
        asia_low = base_price - 0.0010

        asia_mask = (df.index >= asia_start_time) & (df.index <= asia_end_time)
        df.loc[asia_mask, 'Close'] = np.linspace(base_price, asia_high, num=asia_mask.sum())

        grab_time = london_start_time + pd.Timedelta(minutes=15)
        reversal_time = grab_time + pd.Timedelta(minutes=15)
        df.loc[london_start_time, 'Close'] = asia_high + 0.0002
        df.loc[grab_time, 'Close'] = asia_high + 0.0005
        df.loc[reversal_time, 'Close'] = asia_high - 0.0005

    for day in range(15, days, 20): # Bullish pattern
        asia_start_str = f'2023-01-01'
        asia_start_time = pd.to_datetime(asia_start_str) + pd.Timedelta(days=day, hours=0)
        asia_end_time = asia_start_time + pd.Timedelta(hours=7, minutes=45)
        london_start_time = asia_end_time + pd.Timedelta(minutes=15)

        base_price = df.loc[str(asia_start_time.date())].iloc[0]['Close']
        asia_high = base_price + 0.0010
        asia_low = base_price - 0.0010

        asia_mask = (df.index >= asia_start_time) & (df.index <= asia_end_time)
        df.loc[asia_mask, 'Close'] = np.linspace(base_price, asia_low, num=asia_mask.sum())

        grab_time = london_start_time + pd.Timedelta(minutes=15)
        reversal_time = grab_time + pd.Timedelta(minutes=15)
        df.loc[london_start_time, 'Close'] = asia_low - 0.0002
        df.loc[grab_time, 'Close'] = asia_low - 0.0005
        df.loc[reversal_time, 'Close'] = asia_low + 0.0005

    spread = rng.normal(0.0001, 0.00005, size=len(df)) * session_volatility
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.abs(spread)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.abs(spread)
    df = df.drop(columns=['hour']).dropna()
    return df[['Open', 'High', 'Low', 'Close']]

def preprocess_data(df):
    df['hour'] = df.index.hour
    df['day_id'] = pd.factorize(df.index.date)[0]
    df['is_asia'] = (df['hour'] >= 0) & (df['hour'] < 8)
    df['is_london'] = (df['hour'] >= 8) & (df['hour'] < 16)

    # Calculate Asia session high/low for each day
    asia_session_data = df[df['is_asia']].groupby('day_id').agg(
        prev_asia_high=('High', 'max'),
        prev_asia_low=('Low', 'min')
    )

    # Map the calculated values to the *next* day
    asia_session_data = asia_session_data.shift(1)

    df = df.join(asia_session_data, on='day_id')

    # Forward fill the values for the whole day
    df['prev_asia_high'] = df['prev_asia_high'].ffill()
    df['prev_asia_low'] = df['prev_asia_low'].ffill()

    # Calculate Asia range
    df['prev_asia_range'] = df['prev_asia_high'] - df['prev_asia_low']
    df['prev_asia_range_pct'] = (df['prev_asia_range'] / df['prev_asia_low']) * 100

    # Clean up
    df = df.drop(columns=['day_id', 'hour', 'is_asia'])
    df = df.dropna()
    return df

def passthrough(data):
    return data

class AsiaLiquidityGrabReversalStrategy(Strategy):
    asia_range_max_pct = 2.0

    def init(self):
        self.is_london = self.I(passthrough, self.data.df['is_london'].values)
        self.prev_asia_high = self.I(passthrough, self.data.df['prev_asia_high'].values)
        self.prev_asia_low = self.I(passthrough, self.data.df['prev_asia_low'].values)
        self.prev_asia_range_pct = self.I(passthrough, self.data.df['prev_asia_range_pct'].values)
        self.liquidity_grab_up = None
        self.liquidity_grab_down = None

    def is_bearish_engulfing(self):
        if len(self.data.Close) < 2: return False
        prev_open = self.data.Open[-2]
        prev_close = self.data.Close[-2]
        curr_open = self.data.Open[-1]
        curr_close = self.data.Close[-1]
        return (prev_close > prev_open and
                curr_close < curr_open and
                curr_open >= prev_close and # Relaxed condition
                curr_close < prev_open)

    def is_bullish_engulfing(self):
        if len(self.data.Close) < 2: return False
        prev_open = self.data.Open[-2]
        prev_close = self.data.Close[-2]
        curr_open = self.data.Open[-1]
        curr_close = self.data.Close[-1]
        return (prev_close < prev_open and
                curr_close > curr_open and
                curr_open <= prev_close and # Relaxed condition
                curr_close > prev_open)

    def next(self):
        if self.position:
            return

        is_london_session = self.is_london[-1] == 1
        is_valid_asia_range = self.prev_asia_range_pct[-1] < self.asia_range_max_pct

        if not (is_london_session and is_valid_asia_range):
            self.liquidity_grab_up = None
            self.liquidity_grab_down = None
            return

        # Bearish entry logic
        # If a grab happened on the previous candle, check for a reversal on the current one
        if self.liquidity_grab_up:
            if self.is_bearish_engulfing() and self.data.Close[-1] < self.prev_asia_high[-1]:
                sl = self.liquidity_grab_up
                tp = self.prev_asia_low[-1]
                self.sell(sl=sl, tp=tp)
            self.liquidity_grab_up = None  # Reset after one attempt
        # Otherwise, check if a new grab is happening now
        elif self.data.Close[-1] > self.prev_asia_high[-1] and not self.liquidity_grab_down:
            self.liquidity_grab_up = self.data.High[-1]

        # Bullish entry logic
        # If a grab happened on the previous candle, check for a reversal on the current one
        if self.liquidity_grab_down:
            if self.is_bullish_engulfing() and self.data.Close[-1] > self.prev_asia_low[-1]:
                sl = self.liquidity_grab_down
                tp = self.prev_asia_high[-1]
                self.buy(sl=sl, tp=tp)
            self.liquidity_grab_down = None # Reset after one attempt
        # Otherwise, check if a new grab is happening now
        elif self.data.Close[-1] < self.prev_asia_low[-1] and not self.liquidity_grab_up:
            self.liquidity_grab_down = self.data.Low[-1]

if __name__ == '__main__':
    data = generate_forex_data(days=200)
    data = preprocess_data(data)

    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=10000, commission=.002)

    stats = bt.optimize(
        asia_range_max_pct=list(np.arange(0.5, 3.1, 0.5)),
        maximize='Sharpe Ratio'
    )

    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON serialization
    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                sanitized[key] = None # Or handle appropriately
            elif pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (np.int64, np.int32)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                sanitized[key] = float(value)
            else:
                sanitized[key] = value
        return sanitized

    clean_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        # Use a secondary get(key, 0.0) for metrics that can be NaN
        json.dump({
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': clean_stats.get('Return [%]') or 0.0,
            'sharpe': clean_stats.get('Sharpe Ratio') or 0.0,
            'max_drawdown': clean_stats.get('Max. Drawdown [%]') or 0.0,
            'win_rate': clean_stats.get('Win Rate [%]') or 0.0,
            'total_trades': clean_stats.get('# Trades') or 0
        }, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    try:
        bt.plot()
    except TypeError as e:
        print(f"Could not generate plot due to a known issue with the library: {e}")
