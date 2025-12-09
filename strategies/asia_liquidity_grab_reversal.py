import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

def generate_synthetic_data():
    """
    Generates synthetic 15-minute OHLC data tailored for the Asia Liquidity Grab strategy.
    The data includes a clear bearish setup and a clear bullish setup.
    """
    # Define session times in UTC
    asia_start_hour = 0
    asia_end_hour = 8
    uk_start_hour = 9
    uk_end_hour = 17

    # Create a time index for 5 days of 15-minute data
    index = pd.date_range(start='2023-01-02', periods=96 * 5, freq='15min', tz='UTC')
    data = pd.DataFrame(index=index, columns=['Open', 'High', 'Low', 'Close'])

    # --- Day 1: Bearish Setup ---
    day1_start = '2023-01-02 00:00'
    day1_end = '2023-01-02 23:45'
    day1_data = data.loc[day1_start:day1_end].copy()

    # Asia Session (00:00 - 08:00): Low volatility range
    asia_mask = (day1_data.index.hour >= asia_start_hour) & (day1_data.index.hour < asia_end_hour)
    day1_data.loc[asia_mask, 'Open'] = np.linspace(100, 100.1, num=asia_mask.sum())
    day1_data.loc[asia_mask, 'Close'] = np.linspace(100.1, 100.2, num=asia_mask.sum())
    day1_data.loc[asia_mask, 'High'] = day1_data.loc[asia_mask, ['Open', 'Close']].max(axis=1) + 0.05
    day1_data.loc[asia_mask, 'Low'] = day1_data.loc[asia_mask, ['Open', 'Close']].min(axis=1) - 0.05

    asia_high = day1_data.loc[asia_mask, 'High'].max()
    asia_low = day1_data.loc[asia_mask, 'Low'].min()

    # UK Session (09:00 - 17:00): Liquidity grab and reversal
    uk_mask = (day1_data.index.hour >= uk_start_hour) & (day1_data.index.hour < uk_end_hour)

    # Pre-grab candles
    pre_grab_mask = (day1_data.index.hour >= uk_start_hour) & (day1_data.index.hour < 10)
    day1_data.loc[pre_grab_mask, 'Open'] = np.linspace(100.2, 100.3, num=pre_grab_mask.sum())
    day1_data.loc[pre_grab_mask, 'Close'] = np.linspace(100.3, 100.4, num=pre_grab_mask.sum())

    # Grab candle (10:00)
    grab_time = '2023-01-02 10:00'
    day1_data.loc[grab_time, 'Open'] = 100.4
    day1_data.loc[grab_time, 'Close'] = 100.5
    day1_data.loc[grab_time, 'High'] = asia_high + 0.1 # Pierce Asia High
    day1_data.loc[grab_time, 'Low'] = 100.35

    # Engulfing candle (10:15)
    engulf_time = '2023-01-02 10:15'
    day1_data.loc[engulf_time, 'Open'] = 100.55
    day1_data.loc[engulf_time, 'High'] = 100.6
    day1_data.loc[engulf_time, 'Close'] = 100.3 # Engulfs the prior candle's body
    day1_data.loc[engulf_time, 'Low'] = 100.25

    # Post-engulfing trend down towards Asia Low
    post_engulf_mask = (day1_data.index.hour > 10) & uk_mask
    num_post_engulf = post_engulf_mask.sum()
    day1_data.loc[post_engulf_mask, 'Open'] = np.linspace(100.3, asia_low - 0.1, num=num_post_engulf)
    day1_data.loc[post_engulf_mask, 'Close'] = np.linspace(100.2, asia_low - 0.2, num=num_post_engulf)

    day1_data.ffill(inplace=True)
    data.loc[day1_start:day1_end] = day1_data

    # --- Day 2: Bullish Setup ---
    day2_start = '2023-01-03 00:00'
    day2_end = '2023-01-03 23:45'
    day2_data = data.loc[day2_start:day2_end].copy()

    # Asia Session (00:00 - 08:00): Low volatility range
    asia_mask = (day2_data.index.hour >= asia_start_hour) & (day2_data.index.hour < asia_end_hour)
    day2_data.loc[asia_mask, 'Open'] = np.linspace(102, 101.9, num=asia_mask.sum())
    day2_data.loc[asia_mask, 'Close'] = np.linspace(101.9, 101.8, num=asia_mask.sum())
    day2_data.loc[asia_mask, 'High'] = day2_data.loc[asia_mask, ['Open', 'Close']].max(axis=1) + 0.05
    day2_data.loc[asia_mask, 'Low'] = day2_data.loc[asia_mask, ['Open', 'Close']].min(axis=1) - 0.05

    asia_high = day2_data.loc[asia_mask, 'High'].max()
    asia_low = day2_data.loc[asia_mask, 'Low'].min()

    # UK Session (09:00 - 17:00): Liquidity grab and reversal
    uk_mask = (day2_data.index.hour >= uk_start_hour) & (day2_data.index.hour < uk_end_hour)

    # Pre-grab candles
    pre_grab_mask = (day2_data.index.hour >= uk_start_hour) & (day2_data.index.hour < 10)
    day2_data.loc[pre_grab_mask, 'Open'] = np.linspace(101.8, 101.7, num=pre_grab_mask.sum())
    day2_data.loc[pre_grab_mask, 'Close'] = np.linspace(101.7, 101.6, num=pre_grab_mask.sum())

    # Grab candle (10:00)
    grab_time = '2023-01-03 10:00'
    day2_data.loc[grab_time, 'Open'] = 101.6
    day2_data.loc[grab_time, 'Close'] = 101.5
    day2_data.loc[grab_time, 'Low'] = asia_low - 0.1 # Pierce Asia Low
    day2_data.loc[grab_time, 'High'] = 101.65

    # Engulfing candle (10:15)
    engulf_time = '2023-01-03 10:15'
    day2_data.loc[engulf_time, 'Open'] = 101.45
    day2_data.loc[engulf_time, 'Low'] = 101.4
    day2_data.loc[engulf_time, 'Close'] = 101.7 # Engulfs the prior candle's body
    day2_data.loc[engulf_time, 'High'] = 101.75

    # Post-engulfing trend up towards Asia High
    post_engulf_mask = (day2_data.index.hour > 10) & uk_mask
    num_post_engulf = post_engulf_mask.sum()
    day2_data.loc[post_engulf_mask, 'Open'] = np.linspace(101.7, asia_high + 0.1, num=num_post_engulf)
    day2_data.loc[post_engulf_mask, 'Close'] = np.linspace(101.8, asia_high + 0.2, num=num_post_engulf)

    day2_data.ffill(inplace=True)
    data.loc[day2_start:day2_end] = day2_data

    # Fill remaining days with some noise
    data.ffill(inplace=True)
    noise = np.random.normal(0, 0.05, size=data.shape)
    data += noise

    # Ensure OHLC integrity
    data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0.01, 0.05, size=len(data))
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0.01, 0.05, size=len(data))

    return data

def preprocess_data(df):
    """
    Calculates Asia session high, low, and range, and maps them to the entire day.
    """
    df['date'] = df.index.date

    # Define Asia session time
    asia_mask = (df.index.hour >= 0) & (df.index.hour < 8)

    # Group by date and calculate Asia session stats
    asia_stats = df[asia_mask].groupby('date').agg(
        asia_high=('High', 'max'),
        asia_low=('Low', 'min')
    )
    asia_stats['asia_range'] = (asia_stats['asia_high'] - asia_stats['asia_low']) / asia_stats['asia_low'] * 100

    # Map stats to the original dataframe
    df['asia_high'] = df['date'].map(asia_stats['asia_high'])
    df['asia_low'] = df['date'].map(asia_stats['asia_low'])
    df['asia_range'] = df['date'].map(asia_stats['asia_range'])

    # Define UK session
    df['is_uk_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)

    df.dropna(inplace=True)
    return df

class AsiaLiquidityGrabReversalStrategy(Strategy):
    # Optimizable parameters
    max_asia_range = 2.0  # Maximum allowed Asia range in percent
    sl_buffer_pct = 0.1   # Stop loss buffer in percent

    def init(self):
        # State variables to prevent re-entry on the same day
        self.last_trade_date = None

    def next(self):
        # Ensure we have enough data
        if len(self.data.Close) < 2:
            return

        # Get current and previous timestamp
        current_date = self.data.index[-1].date()

        # Only trade during the UK session and once per day
        if not self.data.is_uk_session[-1] or self.position or current_date == self.last_trade_date:
            return

        # --- Strategy Filters ---
        # 1. Check if Asia session data is available
        asia_high = self.data.asia_high[-1]
        asia_low = self.data.asia_low[-1]
        asia_range = self.data.asia_range[-1]

        if pd.isna(asia_high) or pd.isna(asia_low):
            return

        # 2. Asia range filter
        if asia_range > self.max_asia_range:
            return

        # --- Pattern Recognition ---
        # Define the last two candles for pattern checking
        prev_candle_high = self.data.High[-2]
        prev_candle_low = self.data.Low[-2]
        prev_candle_open = self.data.Open[-2]
        prev_candle_close = self.data.Close[-2]

        current_candle_high = self.data.High[-1]
        current_candle_low = self.data.Low[-1]
        current_candle_open = self.data.Open[-1]
        current_candle_close = self.data.Close[-1]

        # --- Bearish Setup (SHORT) ---
        # 1. Liquidity grab: previous candle must have wicked above Asia High
        liquidity_grab_high = prev_candle_high > asia_high

        # 2. Confirmation: current candle must be a bearish engulfing
        is_bearish_engulfing = (current_candle_close < current_candle_open and
                                current_candle_open > prev_candle_close and
                                current_candle_close < prev_candle_open)

        if liquidity_grab_high and is_bearish_engulfing:
            sl = current_candle_high * (1 + self.sl_buffer_pct / 100)
            tp = asia_low

            # Additional check to ensure TP is valid
            if tp < self.data.Close[-1]:
                self.sell(sl=sl, tp=tp)
                self.last_trade_date = current_date


        # --- Bullish Setup (LONG) ---
        # 1. Liquidity grab: previous candle must have wicked below Asia Low
        liquidity_grab_low = prev_candle_low < asia_low

        # 2. Confirmation: current candle must be a bullish engulfing
        is_bullish_engulfing = (current_candle_close > current_candle_open and
                                current_candle_open < prev_candle_close and
                                current_candle_close > prev_candle_open)

        if liquidity_grab_low and is_bullish_engulfing:
            sl = current_candle_low * (1 - self.sl_buffer_pct / 100)
            tp = asia_high

            # Additional check to ensure TP is valid
            if tp > self.data.Close[-1]:
                self.buy(sl=sl, tp=tp)
                self.last_trade_date = current_date

if __name__ == '__main__':
    data = generate_synthetic_data()
    data = preprocess_data(data)

    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=100_000, commission=.002)

    stats = bt.optimize(
        max_asia_range=list(np.arange(1.0, 4.0, 0.5)),
        sl_buffer_pct=list(np.arange(0.1, 0.6, 0.1)),
        maximize='Sharpe Ratio'
    )

    print(stats)

    import os
    os.makedirs('results', exist_ok=True)

    # Helper to convert numpy types to python native types
    def sanitize_json(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: sanitize_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [sanitize_json(e) for e in obj]
        return obj

    # Save results
    with open('results/temp_result.json', 'w') as f:
        stats_dict = stats.to_dict()

        # Ensure all required keys are present
        result_data = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': stats_dict.get('Return [%]', 0.0),
            'sharpe': stats_dict.get('Sharpe Ratio', 0.0),
            'max_drawdown': stats_dict.get('Max. Drawdown [%]', 0.0),
            'win_rate': stats_dict.get('Win Rate [%]', 0.0),
            'total_trades': stats_dict.get('# Trades', 0)
        }

        json.dump(sanitize_json(result_data), f, indent=2)

    print("Successfully saved results to results/temp_result.json")

    # Generate plot
    try:
        bt.plot(filename='results/asia_liquidity_grab_reversal.html', open_browser=False)
        print("Successfully generated plot at results/asia_liquidity_grab_reversal.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
