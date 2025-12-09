import json
import os

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy


class AsiaLiquidityGrabUkReversalStrategy(Strategy):
    # --- Strategy Parameters ---
    asia_range_max_pct = 2.0

    def init(self):
        # All indicators are pre-calculated, so nothing to initialize here.
        # Pre-calculated columns will be available directly on `self.data`.
        pass

    def next(self):
        # Ensure we don't have a position open
        if self.position:
            return

        # Get current values from pre-calculated columns.
        # backtesting.py makes columns from the input DataFrame available here.
        current_session_name = self.data.session[-1]
        is_tradeable = self.data.is_tradeable_range[-1]
        high_of_asia = self.data.HOA[-1]
        low_of_asia = self.data.LOA[-1]
        is_bullish_engulfing = self.data.is_bullish_engulfing[-1]
        is_bearish_engulfing = self.data.is_bearish_engulfing[-1]

        # --- Entry Logic ---
        if current_session_name == 'UK' and is_tradeable:

            # Bearish Setup (Short)
            if self.data.High[-1] > high_of_asia and is_bearish_engulfing:
                sl = self.data.High[-1]
                tp = low_of_asia
                if tp < self.data.Close[-1]:
                    self.sell(sl=sl, tp=tp, size=0.1)

            # Bullish Setup (Long)
            elif self.data.Low[-1] < low_of_asia and is_bullish_engulfing:
                sl = self.data.Low[-1]
                tp = high_of_asia
                if tp > self.data.Close[-1]:
                    self.buy(sl=sl, tp=tp, size=0.1)


if __name__ == '__main__':
    # Since GOOG data is US-based and doesn't have 24h data,
    # it is not suitable for a strategy based on Asia/London sessions.
    # We will generate synthetic Forex data instead.

    # --- Data Generation ---
    def generate_forex_data(days=200):
        """Generates synthetic 24-hour Forex data for backtesting."""
        np.random.seed(42)
        # Generate a 15-minute frequency DatetimeIndex
        index = pd.date_range(start='2023-01-01', periods=days * 24 * 4, freq='15min', tz='UTC')

        # Base price movement with some noise and trend
        base_price = 1.1000
        returns = np.random.randn(len(index)) * 0.0005
        price = base_price + np.cumsum(returns)

        # Add volatility spikes to simulate liquidity grabs
        spike_indices = np.random.choice(len(index), size=int(days * 0.5), replace=False)
        spike_magnitudes = (np.random.rand(len(spike_indices)) - 0.5) * 0.005
        price[spike_indices] += spike_magnitudes

        # Create DataFrame
        df = pd.DataFrame(index=index)
        df['Open'] = price
        df['High'] = df['Open'] + np.random.rand(len(index)) * 0.001
        df['Low'] = df['Open'] - np.random.rand(len(index)) * 0.001
        df['Close'] = df['Open'] + (np.random.rand(len(index)) - 0.5) * 0.002
        df['Volume'] = np.random.randint(100, 1000, size=len(index))

        # Ensure OHLC consistency
        df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
        df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)

        return df

    def preprocess_data(df):
        """Adds session, Asia range, and engulfing pattern indicators."""
        # 1. Identify Sessions
        df['session'] = 'NY'
        df.loc[df.index.hour.isin(range(0, 8)), 'session'] = 'Asia'
        df.loc[df.index.hour.isin(range(8, 17)), 'session'] = 'UK'

        # 2. Calculate Asia Range (HOA/LOA)
        # Group by date and find the high/low of the 'Asia' session for that date
        asia_df = df[df['session'] == 'Asia']
        daily_asia_range = asia_df.groupby(asia_df.index.date)
        hoa = daily_asia_range['High'].max()
        loa = daily_asia_range['Low'].min()

        # Map the HOA and LOA to the entire day
        df['HOA'] = pd.Series(df.index.date, index=df.index).map(hoa)
        df['LOA'] = pd.Series(df.index.date, index=df.index).map(loa)

        # Forward-fill to handle days with no Asia session (weekends)
        df['HOA'] = df['HOA'].ffill()
        df['LOA'] = df['LOA'].ffill()

        # Calculate Asia Range and filter condition
        df['asia_range'] = df['HOA'] - df['LOA']
        df['asia_range_pct'] = (df['asia_range'] / df['LOA']) * 100
        df['is_tradeable_range'] = df['asia_range_pct'] < 2.0

        # 3. Identify Engulfing Patterns
        body_size = abs(df['Close'] - df['Open'])
        prev_body_size = body_size.shift(1)

        is_bullish_engulfing = (
            (df['Close'] > df['Open']) &            # Current is green
            (df['Close'].shift(1) < df['Open'].shift(1)) & # Previous is red
            (df['Open'] < df['Close'].shift(1)) &     # Current open is lower than previous close
            (df['Close'] > df['Open'].shift(1))      # Current close is higher than previous open
        )

        is_bearish_engulfing = (
            (df['Close'] < df['Open']) &             # Current is red
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous is green
            (df['Open'] > df['Close'].shift(1)) &      # Current open is higher than previous close
            (df['Close'] < df['Open'].shift(1))       # Current close is lower than previous open
        )

        df['is_bullish_engulfing'] = is_bullish_engulfing
        df['is_bearish_engulfing'] = is_bearish_engulfing

        return df.dropna()

    data = generate_forex_data()
    data = preprocess_data(data)

    # --- Backtest Execution ---
    bt = Backtest(data, AsiaLiquidityGrabUkReversalStrategy, cash=100_000, commission=.002)

    # --- Optimization ---
    # Note: Using a wide range for optimization can be slow.
    # A smaller, more targeted range is often better.
    stats = bt.optimize(
        asia_range_max_pct=np.arange(1.0, 3.5, 0.5).tolist(),
        maximize='Sharpe Ratio'
    )

    print("Best stats found:")
    print(stats)

    # --- Output Results ---
    os.makedirs('results', exist_ok=True)

    # Check if any trades were made
    if stats['# Trades'] > 0:
        win_rate = stats['Win Rate [%]']
        sharpe = stats['Sharpe Ratio']
    else:
        win_rate = 0
        sharpe = 0

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'asia_liquidity_grab_uk_reversal',
            'return': stats['Return [%]'],
            'sharpe': sharpe,
            'max_drawdown': stats['Max. Drawdown [%]'],
            'win_rate': win_rate,
            'total_trades': stats['# Trades']
        }, f, indent=2)

    print("\nResults saved to results/temp_result.json")

    # Generate plot of the best run
    bt.plot()
