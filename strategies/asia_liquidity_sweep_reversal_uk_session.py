from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data to add session information and daily levels.
    """
    df.index = pd.to_datetime(df.index)

    # Define session times (UTC)
    asia_session_start = pd.to_datetime('00:00').time()
    asia_session_end = pd.to_datetime('08:00').time()
    uk_session_start = pd.to_datetime('08:00').time()
    uk_session_end = pd.to_datetime('16:00').time()

    # Identify sessions
    df['is_asia'] = (df.index.time >= asia_session_start) & (df.index.time < asia_session_end)
    df['is_uk'] = (df.index.time >= uk_session_start) & (df.index.time < uk_session_end)

    # Calculate daily groups for session calculations
    df['date'] = df.index.date

    # Calculate HOA and LOA
    asia_high = df[df['is_asia']].groupby('date')['High'].max()
    asia_low = df[df['is_asia']].groupby('date')['Low'].min()

    df['HOA'] = df['date'].map(asia_high)
    df['LOA'] = df['date'].map(asia_low)

    df['HOA'] = df['HOA'].ffill()
    df['LOA'] = df['LOA'].ffill()

    df['asia_range'] = (df['HOA'] - df['LOA']) / df['LOA'] * 100

    return df

class AsiaLiquiditySweepReversalUkSessionStrategy(Strategy):
    asia_range_max = 2.0

    def init(self):
        pass

    def next(self):
        # Only trade during UK session
        if not self.data.is_uk[-1]:
            return

        # Check if we are in a new bar of the UK session
        if not self.data.is_uk[-2] and self.data.is_uk[-1]:
             # Ensure we have valid HOA/LOA from Asia session
            if pd.isna(self.data.HOA[-1]) or pd.isna(self.data.LOA[-1]):
                return

        # Ensure Asia range is within the filter
        if self.data.asia_range[-1] > self.asia_range_max:
            return

        # --- SHORT ENTRY ---
        # 1. Price broke above HOA
        # 2. Bearish engulfing reversal below HOA
        if (self.data.High[-2] > self.data.HOA[-2] and
            self.data.Close[-1] < self.data.HOA[-1] and
            self.data.Close[-1] < self.data.Open[-1] and # Bearish candle
            self.data.Open[-1] > self.data.Close[-2] and # Engulfing
            self.data.Close[-1] < self.data.Open[-2]):

            sl = self.data.High[-1]
            tp = self.data.LOA[-1]
            self.sell(sl=sl, tp=tp, size=1)

        # --- LONG ENTRY ---
        # 1. Price broke below LOA
        # 2. Bullish engulfing reversal above LOA
        elif (self.data.Low[-2] < self.data.LOA[-2] and
              self.data.Close[-1] > self.data.LOA[-1] and
              self.data.Close[-1] > self.data.Open[-1] and # Bullish candle
              self.data.Close[-1] > self.data.Open[-2] and # Engulfing
              self.data.Open[-1] < self.data.Close[-2]):

            sl = self.data.Low[-1]
            tp = self.data.HOA[-1]
            self.buy(sl=sl, tp=tp, size=1)


def generate_forex_data(days=100):
    """Generates synthetic 15-minute forex data."""
    n_bars = days * 24 * 4  # 15-minute bars in a day
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='15min')

    price = 1.1000
    returns = np.random.normal(loc=0, scale=0.0005, size=n_bars)
    price_path = price * (1 + returns).cumprod()

    df = pd.DataFrame(index=dates)
    df['Open'] = price_path
    df['High'] = df['Open'] + np.random.uniform(0, 0.001, size=n_bars)
    df['Low'] = df['Open'] - np.random.uniform(0, 0.001, size=n_bars)
    df['Close'] = df['Open'] + np.random.normal(loc=0, scale=0.0005, size=n_bars)
    df['Volume'] = np.random.randint(100, 1000, size=n_bars)

    # Ensure OHLC consistency
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 0.0005, size=n_bars)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 0.0005, size=n_bars)

    return df

if __name__ == '__main__':
    # Generate and preprocess data
    data = generate_forex_data(days=50) # Using 50 days to keep optimization fast
    data = preprocess_data(data)

    # Run backtest
    bt = Backtest(data, AsiaLiquiditySweepReversalUkSessionStrategy, cash=100_000, commission=.0002)

    # Optimize
    stats = bt.optimize(
        asia_range_max=np.arange(0.5, 4.0, 0.5).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.asia_range_max > 0
    )

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Handle cases with no trades
    win_rate = stats.get('Win Rate [%]', 0)
    sharpe = stats.get('Sharpe Ratio', 0)

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'asia_liquidity_sweep_reversal_uk_session',
            'return': float(stats.get('Return [%]', 0)),
            'sharpe': float(sharpe) if not np.isnan(sharpe) else None,
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)),
            'win_rate': float(win_rate) if not np.isnan(win_rate) else None,
            'total_trades': int(stats.get('# Trades', 0))
        }, f, indent=2)

    # Generate plot
    bt.plot(filename='results/asia_liquidity_sweep_reversal_uk_session.html')
