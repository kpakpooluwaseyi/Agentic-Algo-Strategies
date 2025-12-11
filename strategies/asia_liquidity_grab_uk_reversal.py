from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
import json
import os

class AsiaLiquidityGrabUkReversalStrategy(Strategy):
    # --- Strategy Parameters ---
    asia_range_max_pct = 2.0  # Max Asia session range in percent

    # --- State Tracking Variables ---
    def init(self):
        # These are not indicators in the traditional sense, but convenient
        # shortcuts to access pre-calculated data columns.
        self.asia_high = self.I(lambda x: x, self.data.df['asia_high'])
        self.asia_low = self.I(lambda x: x, self.data.df['asia_low'])
        self.asia_range_pct = self.I(lambda x: x, self.data.df['asia_range_pct'])
        self.is_uk_session = self.I(lambda x: x, self.data.df['is_uk_session'])

        # State machine to track the setup
        self.liquidity_grabbed = False
        self.liquidity_grab_high = 0.0  # Track the highest point of the grab

    def next(self):
        # --- Pre-computation for the current bar ---
        current_high = self.data.High[-1]
        current_close = self.data.Close[-1]

        # --- State Machine & Entry Logic ---

        # 1. Reset state daily
        # If it's the first bar of a new day, reset the state.
        if len(self.data) > 1 and self.data.index[-1].date() != self.data.index[-2].date():
            self.liquidity_grabbed = False
            self.liquidity_grab_high = 0.0

        # 2. Check for Liquidity Grab and track the peak of the formation
        is_potential_grab_bar = (
            self.is_uk_session[-1] and
            self.asia_range_pct[-1] < self.asia_range_max_pct and
            current_high > self.asia_high[-1]
        )

        if not self.position:
            # If a grab starts, set the flag and record the initial high.
            if not self.liquidity_grabbed and is_potential_grab_bar:
                self.liquidity_grabbed = True
                self.liquidity_grab_high = current_high
            # If a grab is already in progress, update the high water mark.
            elif self.liquidity_grabbed:
                self.liquidity_grab_high = max(self.liquidity_grab_high, current_high)

        # 3. Check for Reversal Confirmation (Bearish Engulfing)
        # Conditions:
        # - The liquidity grab has happened.
        # - We are not in a position.
        if self.liquidity_grabbed and not self.position:
            # Bearish Engulfing check (more robust):
            # - Current close is below the previous open.
            # - Current open is above the previous close.
            # - Current body engulfs the previous body (using abs).
            is_engulfing = (
                current_close < self.data.Open[-2] and
                self.data.Open[-1] > self.data.Close[-2] and
                (self.data.Open[-1] - current_close) > abs(self.data.Close[-2] - self.data.Open[-2])
            )

            if is_engulfing:
                # --- Execute Trade ---
                # Place stop loss above the HIGHEST high of the formation
                sl = self.liquidity_grab_high + 0.0005  # Small buffer

                # Place take profit at the Asia session low
                tp = self.asia_low[-1]

                # Ensure SL is above entry and TP is below entry
                if sl > current_close and tp < current_close:
                    self.sell(sl=sl, tp=tp)

                # Reset state after taking the trade
                self.liquidity_grabbed = False
                self.liquidity_grab_high = 0.0

if __name__ == '__main__':
    # --- Data Generation and Preprocessing ---
    def generate_forex_data(days=90):
        """Generates synthetic 24-hour Forex data for backtesting."""
        rng = np.random.default_rng(42)

        # Generate 15-minute timestamps for the specified number of days
        # 96 periods per day (24 hours * 4)
        timestamps = pd.to_datetime(pd.date_range('2023-01-01', periods=days * 96, freq='15min', tz='UTC'))

        # Base price movement with some randomness
        price = 1.1000
        returns = rng.normal(loc=0, scale=0.0003, size=len(timestamps))
        prices = price * np.exp(np.cumsum(returns))

        # Create DataFrame
        df = pd.DataFrame(index=timestamps, data={'Close': prices})

        # Simulate session-based volatility
        # Make Asia session low volatility
        asia_mask = (df.index.hour >= 0) & (df.index.hour < 8)
        df.loc[asia_mask, 'Close'] += rng.normal(loc=0, scale=0.0001, size=asia_mask.sum())

        # Make UK session high volatility (for liquidity grabs)
        uk_mask = (df.index.hour >= 8) & (df.index.hour < 16)
        df.loc[uk_mask, 'Close'] += rng.normal(loc=0, scale=0.0005, size=uk_mask.sum())

        # Generate OHLC data
        df['Open'] = df['Close'].shift(1)
        df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0.0001, 0.0005, size=len(df))
        df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0.0001, 0.0005, size=len(df))

        # Ensure consistency
        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)

        return df.dropna()

    def preprocess_data(df):
        """Calculates Asia session metrics needed for the strategy."""
        df['date'] = df.index.date

        # Define session times in UTC
        asia_session_mask = (df.index.hour >= 0) & (df.index.hour < 8)

        # Calculate daily Asia session high and low
        asia_high = df[asia_session_mask].groupby('date')['High'].max()
        asia_low = df[asia_session_mask].groupby('date')['Low'].min()

        # Map daily values back to the main dataframe
        df['asia_high'] = df['date'].map(asia_high)
        df['asia_low'] = df['date'].map(asia_low)

        # Calculate Asia range and percentage
        asia_range = asia_high - asia_low
        df['asia_range_pct'] = (asia_range / asia_low) * 100

        # Define UK session for trade entry
        df['is_uk_session'] = (df.index.hour >= 8) & (df.index.hour < 16)

        return df.dropna()

    data = generate_forex_data(days=180)
    data = preprocess_data(data)

    # Run backtest
    bt = Backtest(data, AsiaLiquidityGrabUkReversalStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(
        asia_range_max_pct=np.arange(0.5, 3.0, 0.5),
        maximize='Sharpe Ratio'
    )

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'asia_liquidity_grab_uk_reversal',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    # Generate plot
    bt.plot()
