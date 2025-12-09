
import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply


def generate_synthetic_data():
    """
    Generates synthetic 24-hour data with a specific Asia liquidity grab pattern.
    """
    # Base timeframe
    dates = pd.to_datetime(pd.date_range('2023-01-01 00:00', '2023-01-10 23:45', freq='15min'))
    n = len(dates)

    # Generate random walk data
    base_price = 1.0
    returns = np.random.randn(n) * 0.001
    price = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(index=dates)
    df['Open'] = price
    df['High'] = df['Open'] + np.random.uniform(0, 0.001, n)
    df['Low'] = df['Open'] - np.random.uniform(0, 0.001, n)
    df['Close'] = df['Open'] + (df['High'] - df['Low']) * (np.random.random(n) - 0.5)
    df['Volume'] = np.random.randint(100, 1000, n)

    # --- Craft a specific pattern on a given day ---
    pattern_day = '2023-01-05'

    # 1. Define Asia Session (00:00-08:00) with clear range
    asia_mask = (df.index.date == pd.to_datetime(pattern_day).date()) & (df.index.hour >= 0) & (df.index.hour < 8)
    asia_high_price = 1.01
    asia_low_price = 1.00

    df.loc[asia_mask, 'High'] = np.linspace(asia_low_price, asia_high_price, asia_mask.sum()) + 0.0005
    df.loc[asia_mask, 'Low'] = np.linspace(asia_low_price, asia_high_price, asia_mask.sum()) - 0.0005
    df.loc[asia_mask, 'Open'] = df.loc[asia_mask, 'Low'] + 0.0001
    df.loc[asia_mask, 'Close'] = df.loc[asia_mask, 'High'] - 0.0001

    # Ensure High/Low are correct
    df.loc[df.index == f'{pattern_day} 07:45:00', 'High'] = asia_high_price
    df.loc[df.index == f'{pattern_day} 01:00:00', 'Low'] = asia_low_price


    # 2. UK Session (08:00-17:00) Liquidity Grab
    uk_session_start = f'{pattern_day} 08:00:00'
    grab_time = f'{pattern_day} 09:00:00'

    # Price action leading to the grab
    df.loc[uk_session_start:grab_time, 'Close'] = np.linspace(df.loc[f'{pattern_day} 07:45:00', 'Close'], asia_high_price + 0.001, 5)
    for col in ['Open', 'Low', 'High']:
        df.loc[uk_session_start:grab_time, col] = df.loc[uk_session_start:grab_time, 'Close']

    # The grab candle
    df.loc[grab_time, 'High'] = asia_high_price + 0.002  # Clear grab
    df.loc[grab_time, 'Open'] = asia_high_price + 0.0005
    df.loc[grab_time, 'Close'] = asia_high_price + 0.001
    df.loc[grab_time, 'Low'] = asia_high_price

    # 3. Reversal Confirmation (Bearish Engulfing)
    reversal_time = f'{pattern_day} 09:15:00'

    # Previous candle is bullish (the grab candle)
    df.loc[grab_time, 'Close'] = df.loc[grab_time, 'Open'] + 0.0005

    # Reversal candle
    df.loc[reversal_time, 'Open'] = df.loc[grab_time, 'Close'] + 0.0001 # Opens higher
    df.loc[reversal_time, 'High'] = df.loc[reversal_time, 'Open'] + 0.0005
    df.loc[reversal_time, 'Close'] = df.loc[grab_time, 'Open'] - 0.0005 # Closes lower, engulfing
    df.loc[reversal_time, 'Low'] = df.loc[reversal_time, 'Close'] - 0.0001

    # Ensure it closes back below Asia High
    df.loc[reversal_time, 'Close'] = asia_high_price - 0.0005

    return df


def preprocess_data(df):
    """
    Calculates session-based features for the strategy.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Define sessions
    df['is_asia'] = (df.index.hour >= 0) & (df.index.hour < 8)
    df['is_uk'] = (df.index.hour >= 8) & (df.index.hour < 17)

    # Calculate daily Asia session stats
    asia_session_data = df[df['is_asia']].copy()
    daily_asia_high = asia_session_data.groupby(asia_session_data.index.date)['High'].max()
    daily_asia_low = asia_session_data.groupby(asia_session_data.index.date)['Low'].min()

    daily_asia_range = (daily_asia_high - daily_asia_low) / daily_asia_low * 100

    # Map daily stats back to the original dataframe
    df['asia_high'] = df.index.map(lambda x: daily_asia_high.get(x.date(), np.nan))
    df['asia_low'] = df.index.map(lambda x: daily_asia_low.get(x.date(), np.nan))
    df['asia_range_pct'] = df.index.map(lambda x: daily_asia_range.get(x.date(), np.nan))

    # Forward fill the values to be available throughout the day
    df[['asia_high', 'asia_low', 'asia_range_pct']] = df[['asia_high', 'asia_low', 'asia_range_pct']].fillna(method='ffill')

    df.dropna(inplace=True)
    return df

# Simple pass-through function for backtesting.py to accept pre-calculated columns
def pass_through(series):
    return series

class AsiaLiquidityGrabReversalUkSessionStrategy(Strategy):
    sl_buffer_pct = 0.01

    def init(self):
        # Make pre-calculated columns available to the strategy
        self.is_uk = self.I(pass_through, self.data.df['is_uk'].values, name="is_uk")
        self.asia_high = self.I(pass_through, self.data.df['asia_high'].values, name="asia_high")
        self.asia_low = self.I(pass_through, self.data.df['asia_low'].values, name="asia_low")
        self.asia_range_pct = self.I(pass_through, self.data.df['asia_range_pct'].values, name="asia_range_pct")

        # State variables
        self.grab_occurred_today = False
        self.traded_today = False
        self.current_day = -1


    def next(self):
        # --- Daily State Reset ---
        today = self.data.index[-1].day
        if self.current_day != today:
            self.current_day = today
            self.grab_occurred_today = False
            self.traded_today = False

        # --- Strategy Filters ---
        if self.traded_today or self.position:
            return

        if not self.is_uk[-1]:
            return

        # Asia Range Filter: If range > 2%, disable for the day
        if self.asia_range_pct[-1] > 2.0:
            self.traded_today = True # Effectively disables trading
            return

        # --- State Machine Logic ---

        # 1. Detect Liquidity Grab
        if self.data.High[-1] > self.asia_high[-1]:
            self.grab_occurred_today = True

        # 2. Look for Reversal Confirmation (after grab)
        if self.grab_occurred_today:
            # Bearish Engulfing Pattern
            is_prev_bullish = self.data.Close[-2] > self.data.Open[-2]
            is_curr_bearish = self.data.Close[-1] < self.data.Open[-1]
            is_engulfing = (self.data.Open[-1] > self.data.Close[-2] and
                            self.data.Close[-1] < self.data.Open[-2])

            # Reversal Confirmation: engulfing candle closes back below Asia High
            if is_prev_bullish and is_curr_bearish and is_engulfing and (self.data.Close[-1] < self.asia_high[-1]):

                # --- Entry and Risk Management ---
                sl = self.data.High[-1] * (1 + self.sl_buffer_pct / 100)
                tp = self.asia_low[-1]

                # Additional check to ensure valid SL/TP
                if self.data.Close[-1] > tp and self.data.Close[-1] < sl:
                    self.sell(sl=sl, tp=tp)
                    self.traded_today = True


if __name__ == '__main__':
    data = generate_synthetic_data()
    processed_data = preprocess_data(data)

    if processed_data.empty:
        raise ValueError("Data preprocessing resulted in an empty DataFrame. Check session logic.")

    bt = Backtest(processed_data, AsiaLiquidityGrabReversalUkSessionStrategy, cash=100000, commission=.002)

    stats = bt.optimize(
        sl_buffer_pct=np.arange(0.01, 0.1, 0.02).tolist(),
        maximize='Sharpe Ratio'
    )

    print(stats)

    # --- Output Requirements ---
    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON serialization
    results_dict = {
        'strategy_name': 'asia_liquidity_grab_reversal_uk_session',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    for key, value in results_dict.items():
        if isinstance(value, (np.integer, np.int64)):
            results_dict[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            results_dict[key] = float(value)

    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("Saved results to results/temp_result.json")

    try:
        bt.plot()
        print("Generated plot.")
    except Exception as e:
        print(f"Could not generate plot: {e}")
