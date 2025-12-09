from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os
from datetime import time

def generate_synthetic_data(days=500):
    """
    Generates synthetic 24-hour forex data for backtesting.
    """
    rng = pd.date_range('2022-01-01', periods=days * 96, freq='15min', tz='UTC')

    np.random.seed(42)
    price_movements = np.random.randn(len(rng), 1).cumsum() * 0.001
    volatility = np.random.uniform(0.0001, 0.0005, len(rng))

    base_price = 1.1000
    close_prices = base_price + price_movements

    df = pd.DataFrame(index=rng)
    df['Close'] = close_prices
    df['Open'] = df['Close'].shift(1)
    df['High'] = df[['Open', 'Close']].max(axis=1) + volatility
    df['Low'] = df[['Open', 'Close']].min(axis=1) - volatility
    df.iloc[0, df.columns.get_loc('Open')] = base_price
    df.dropna(inplace=True)

    return df[['Open', 'High', 'Low', 'Close']]

def preprocess_data(df):
    """
    Identifies trading sessions and calculates Asia session high/low.
    """
    df_copy = df.copy()

    asia_start = time(0, 0)
    asia_end = time(8, 0)
    uk_start = time(7, 0)
    uk_end = time(16, 0)

    df_copy['is_asia'] = (df_copy.index.time >= asia_start) & (df_copy.index.time < asia_end)
    df_copy['is_uk'] = (df_copy.index.time >= uk_start) & (df_copy.index.time < uk_end)

    df_copy['session_id'] = df_copy.index.date

    asia_session_data = df_copy[df_copy['is_asia']]
    daily_asia_stats = asia_session_data.groupby('session_id').agg(
        asia_high=('High', 'max'),
        asia_low=('Low', 'min')
    )

    df_copy['asia_high'] = df_copy['session_id'].map(daily_asia_stats['asia_high'])
    df_copy['asia_low'] = df_copy['session_id'].map(daily_asia_stats['asia_low'])

    df_copy[['asia_high', 'asia_low']] = df_copy[['asia_high', 'asia_low']].ffill()

    df_copy['asia_range'] = df_copy['asia_high'] - df_copy['asia_low']
    df_copy['asia_range_pct'] = (df_copy['asia_range'] / df_copy['asia_low']) * 100

    df_copy.dropna(inplace=True)

    return df_copy

# Custom indicator function to pass pre-calculated data
def identity(data, **kwargs):
    return data

class AsiaLiquidityGrabReversalUkSessionStrategy(Strategy):
    asia_range_max_pct = 2.0

    def init(self):
        self.is_uk = self.I(identity, self.data.df['is_uk'])
        self.asia_high = self.I(identity, self.data.df['asia_high'])
        self.asia_low = self.I(identity, self.data.df['asia_low'])
        self.asia_range_pct = self.I(identity, self.data.df['asia_range_pct'])

        self.grab_state = 0 # 0: Neutral, 1: AHi grabbed, -1: ALo grabbed

    def next(self):
        # Only trade during UK session and if Asia range is within limits
        if not self.is_uk[-1] or self.asia_range_pct[-1] > self.asia_range_max_pct:
             # Reset grab state if outside UK session
            if not self.is_uk[-1]:
                self.grab_state = 0
            return

        # ---- Short Entry Logic ----
        # 1. Detect grab of Asia High
        if self.data.High[-1] > self.asia_high[-1] and self.data.Close[-1] < self.asia_high[-1]:
            self.grab_state = 1

        # 2. Check for bearish engulfing confirmation
        if self.grab_state == 1:
            is_bearish_engulfing = (self.data.Close[-1] < self.data.Open[-1] and
                                    self.data.Open[-1] > self.data.Close[-2] and
                                    self.data.Close[-1] < self.data.Open[-2])

            if is_bearish_engulfing and not self.position:
                sl = self.data.High[-1] * 1.0005
                tp = self.asia_low[-1]

                # R:R check
                if (self.data.Close[-1] - sl) != 0 and (tp - self.data.Close[-1]) != 0:
                    if (self.data.Close[-1] - sl) / (tp - self.data.Close[-1]) < -1.0: # Simplified 1:1
                        self.sell(sl=sl, tp=tp)
                self.grab_state = 0

        # ---- Long Entry Logic ----
        # 1. Detect grab of Asia Low
        if self.data.Low[-1] < self.asia_low[-1] and self.data.Close[-1] > self.asia_low[-1]:
            self.grab_state = -1

        # 2. Check for bullish engulfing confirmation
        if self.grab_state == -1:
            is_bullish_engulfing = (self.data.Close[-1] > self.data.Open[-1] and
                                    self.data.Open[-1] < self.data.Close[-2] and
                                    self.data.Close[-1] > self.data.Open[-2])

            if is_bullish_engulfing and not self.position:
                sl = self.data.Low[-1] * 0.9995
                tp = self.asia_high[-1]

                # R:R check
                if (self.data.Close[-1] - sl) != 0 and (tp - self.data.Close[-1]) != 0:
                    if (self.data.Close[-1] - sl) / (tp - self.data.Close[-1]) > 1.0: # Simplified 1:1
                         self.buy(sl=sl, tp=tp)
                self.grab_state = 0

if __name__ == '__main__':
    data = generate_synthetic_data(days=250)
    data = preprocess_data(data)

    bt = Backtest(data, AsiaLiquidityGrabReversalUkSessionStrategy, cash=100000, commission=.0002)

    stats = bt.optimize(
        asia_range_max_pct=list(np.arange(1.0, 3.5, 0.5)),
        maximize='Sharpe Ratio'
    )

    os.makedirs('results', exist_ok=True)

    sanitized_stats = {
        'strategy_name': 'asia_liquidity_grab_reversal_uk_session',
        'return': float(stats.get('Return [%]', 0.0)),
        'sharpe': float(stats.get('Sharpe Ratio', 0.0)),
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
        'win_rate': float(stats.get('Win Rate [%]', 0.0)),
        'total_trades': int(stats.get('# Trades', 0))
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=2)

    try:
        bt.plot(filename="results/asia_liquidity_grab.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
