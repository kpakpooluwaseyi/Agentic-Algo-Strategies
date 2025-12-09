
import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import resample_apply

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index)

    # Identify sessions
    df['session'] = 'Other'
    df.loc[df.index.hour.isin(range(0, 8)), 'session'] = 'Asia'
    df.loc[df.index.hour.isin(range(8, 17)), 'session'] = 'UK'

    # Calculate daily Asia session high and low
    asia_session_data = df[df['session'] == 'Asia']
    daily_asia_high = asia_session_data.groupby(asia_session_data.index.date)['High'].max()
    daily_asia_low = asia_session_data.groupby(asia_session_data.index.date)['Low'].min()

    df['HOA'] = df.index.date
    df['HOA'] = df['HOA'].map(daily_asia_high)
    df['HOA'] = df['HOA'].ffill()

    df['LOA'] = df.index.date
    df['LOA'] = df['LOA'].map(daily_asia_low)
    df['LOA'] = df['LOA'].ffill()

    df['asia_range'] = (df['HOA'] - df['LOA']) / df['LOA'] * 100

    # Candlestick patterns
    # Bearish Engulfing
    df['bearish_engulfing'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] > df['Close']) &
        (df['Open'] >= df['Close'].shift(1)) &
        (df['Close'] <= df['Open'].shift(1)) &
        ((df['Open'] - df['Close']) > (df['Close'].shift(1) - df['Open'].shift(1)))
    )

    # Bullish Engulfing
    df['bullish_engulfing'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Open'] < df['Close']) &
        (df['Open'] <= df['Close'].shift(1)) &
        (df['Close'] >= df['Open'].shift(1)) &
        ((df['Close'] - df['Open']) > (df['Open'].shift(1) - df['Close'].shift(1)))
    )

    df = df.dropna()
    return df

class AsiaLiquidityGrabReversalUkSessionStrategy(Strategy):
    asia_range_max = 2.0

    def init(self):
        self.current_day = None

    def next(self):
        # Ensure we don't trade on the same day a signal has already been acted upon.
        today = self.data.index[-1].date()
        if self.current_day == today and self.position:
            return

        # Only trade during the UK session
        if self.data.session[-1] != 'UK':
            return

        # Asia Range Filter
        if self.data.asia_range[-1] > self.asia_range_max:
            return

        # Short Entry
        if (self.data.High[-2] > self.data.HOA[-2] and
            self.data.bearish_engulfing[-1] and
            not self.position):

            sl = self.data.High[-1]
            tp = self.data.LOA[-1]

            if tp < self.data.Close[-1]:
                self.sell(sl=sl, tp=tp)
                self.current_day = today

        # Long Entry
        if (self.data.Low[-2] < self.data.LOA[-2] and
            self.data.bullish_engulfing[-1] and
            not self.position):

            sl = self.data.Low[-1]
            tp = self.data.HOA[-1]

            if tp > self.data.Close[-1]:
                self.buy(sl=sl, tp=tp)
                self.current_day = today

if __name__ == '__main__':
    from backtesting import Backtest
    import json
    import os

    def generate_synthetic_data(days=180):
        """Generates synthetic 24-hour FX-style data."""
        n = days * 24 * 4  # 15-min intervals
        dates = pd.date_range(start='2023-01-01', periods=n, freq='15min')

        # Base random walk for price
        price = 1.1000 + np.random.randn(n).cumsum() * 0.0001

        # Introduce some volatility and patterns
        price += np.sin(np.linspace(0, 200, n)) * 0.005

        # Create OHLC
        df = pd.DataFrame(index=dates)
        df['Open'] = price
        df['High'] = df['Open'] + np.random.uniform(0, 0.001, n)
        df['Low'] = df['Open'] - np.random.uniform(0, 0.001, n)
        df['Close'] = df['Open'] + np.random.uniform(-0.0005, 0.0005, n)

        # Ensure H >= max(O,C) and L <= min(O,C)
        df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 0.0005, n)
        df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 0.0005, n)

        return df

    # Generate and preprocess data
    data = generate_synthetic_data(days=180)
    data = preprocess_data(data)

    # Initialize backtest
    bt = Backtest(data, AsiaLiquidityGrabReversalUkSessionStrategy, cash=100000, commission=.0002)

    # Optimize
    stats = bt.optimize(
        asia_range_max=np.arange(0.5, 3.5, 0.5).tolist(),
        maximize='Sharpe Ratio'
    )

    # Save results
    os.makedirs('results', exist_ok=True)

    # Handle cases where no trades were made
    if stats['# Trades'] > 0:
        win_rate = float(stats['Win Rate [%]'])
        sharpe = float(stats['Sharpe Ratio'])
    else:
        win_rate = 0.0
        sharpe = 0.0

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'asia_liquidity_grab_reversal_uk_session',
            'return': float(stats['Return [%]']),
            'sharpe': sharpe,
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': win_rate,
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    # Generate plot
    bt.plot()
