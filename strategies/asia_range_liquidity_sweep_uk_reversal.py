import pandas as pd
from backtesting import Strategy
from backtesting.lib import resample_apply

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame to add session information and Asia range data.
    - Defines Asia (00:00-08:00 UTC) and UK (08:00-16:00 UTC) sessions.
    - Calculates the High, Low, and percentage range of the Asia session for each day.
    - Maps these daily values back to each 15-minute bar.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")

    df = df.copy()

    # Define sessions based on UTC time
    df['session'] = 'Other'
    df.loc[df.index.hour < 8, 'session'] = 'Asia'
    df.loc[(df.index.hour >= 8) & (df.index.hour < 16), 'session'] = 'UK'

    # Group by date to calculate daily Asia session stats
    asia_session_data = df[df['session'] == 'Asia'].groupby(df.index.date)

    daily_asia_high = asia_session_data['High'].max()
    daily_asia_low = asia_session_data['Low'].min()
    daily_asia_open = asia_session_data['Open'].first()

    # Create a daily DataFrame and then map it to the original index
    daily_stats = pd.DataFrame({
        'asia_high': daily_asia_high,
        'asia_low': daily_asia_low,
        'asia_open': daily_asia_open,
    })

    daily_stats['asia_range'] = daily_stats['asia_high'] - daily_stats['asia_low']
    daily_stats['asia_range_pct'] = (daily_stats['asia_range'] / daily_stats['asia_open']) * 100

    # Map the daily stats to the main DataFrame's index
    df['asia_high'] = df.index.normalize().map(daily_stats['asia_high'])
    df['asia_low'] = df.index.normalize().map(daily_stats['asia_low'])
    df['asia_range_pct'] = df.index.normalize().map(daily_stats['asia_range_pct'])

    # Forward-fill the mapped data to apply it throughout the day
    df[['asia_high', 'asia_low', 'asia_range_pct']] = df[['asia_high', 'asia_low', 'asia_range_pct']].ffill()

    return df

class AsiaRangeLiquiditySweepUkReversalStrategy(Strategy):
    """
    Implements the Asia Range Liquidity Sweep strategy.
    Enters a trade after a liquidity sweep of the Asia session high/low
    during the UK session, confirmed by an engulfing candle.
    """
    max_asia_range_pct = 2.0  # Max Asia range in percent
    risk_pct = 1.0              # Percentage of equity to risk per trade

    def init(self):
        # State variables to track the setup across candles
        self.sweep_high = None
        self.sweep_low = None
        self.asia_high_swept = False
        self.asia_low_swept = False
        self.current_day = None

    def next(self):
        # Reset state at the beginning of a new day
        if self.data.index[-1].date() != self.current_day:
            self.current_day = self.data.index[-1].date()
            self.asia_high_swept = False
            self.asia_low_swept = False
            self.sweep_high = None
            self.sweep_low = None

        # Strategy only runs during the UK session and if a valid Asia range exists
        if self.data.session[-1] != 'UK' or pd.isna(self.data.asia_high[-1]):
            return

        # Pre-condition: Asia range must be within the defined limit
        if self.data.asia_range_pct[-1] > self.max_asia_range_pct:
            return

        # --- SHORT ENTRY LOGIC ---
        # 1. Trigger: Price breaks above the Asia High
        if not self.asia_high_swept and self.data.High[-1] > self.data.asia_high[-1]:
            self.asia_high_swept = True
            self.sweep_high = self.data.High[-1]  # Record the high of the sweep candle
            self.asia_low_swept = False # Invalidate any long setup

        # 2. Confirmation: A bearish engulfing candle forms after the sweep
        if self.asia_high_swept:
            is_bearish_engulfing = (self.data.Close[-1] < self.data.Open[-1] and # Bearish candle
                                    self.data.Open[-1] >= self.data.Close[-2] and # Engulfs previous body
                                    self.data.Close[-1] <= self.data.Open[-2] and
                                    self.data.Close[-1] < self.data.asia_high[-1]) # Closes back below Asia High

            if is_bearish_engulfing:
                sl = self.sweep_high
                tp = self.data.asia_low[-1]

                # Ensure TP is valid (below current price)
                if tp < self.data.Close[-1]:
                    size = (self.equity * self.risk_pct / 100) / (sl - self.data.Close[-1])
                    if size > 0:
                        self.sell(sl=sl, tp=tp, size=max(1, int(size))) # Ensure size is at least 1

                self.asia_high_swept = False # Reset after trade

        # --- LONG ENTRY LOGIC (INVERSE) ---
        # 1. Trigger: Price breaks below the Asia Low
        if not self.asia_low_swept and self.data.Low[-1] < self.data.asia_low[-1]:
            self.asia_low_swept = True
            self.sweep_low = self.data.Low[-1] # Record the low of the sweep candle
            self.asia_high_swept = False # Invalidate any short setup

        # 2. Confirmation: A bullish engulfing candle forms after the sweep
        if self.asia_low_swept:
            is_bullish_engulfing = (self.data.Close[-1] > self.data.Open[-1] and # Bullish candle
                                    self.data.Close[-1] >= self.data.Open[-2] and # Engulfs previous body
                                    self.data.Open[-1] <= self.data.Close[-2] and
                                    self.data.Close[-1] > self.data.asia_low[-1]) # Closes back above Asia Low

            if is_bullish_engulfing:
                sl = self.sweep_low
                tp = self.data.asia_high[-1]

                # Ensure TP is valid (above current price)
                if tp > self.data.Close[-1]:
                    size = (self.equity * self.risk_pct / 100) / (self.data.Close[-1] - sl)
                    if size > 0:
                        self.buy(sl=sl, tp=tp, size=max(1, int(size))) # Ensure size is at least 1

                self.asia_low_swept = False # Reset after trade

if __name__ == '__main__':
    import numpy as np
    import json
    import os
    from backtesting import Backtest

    def generate_synthetic_data(days=200):
        """Generates synthetic 24-hour OHLCV data for backtesting forex strategies."""
        rng = np.random.default_rng(seed=42)
        n_points = days * 24 * 4  # 15-minute intervals

        # Create a datetime index
        index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_points, freq='15min', tz='UTC'))

        # Generate price movements with some trend and noise
        price = 1.0 + np.cumsum(rng.normal(0, 0.001, n_points))

        # Create OHLC data
        open_price = price
        close_price = open_price + rng.normal(0, 0.0005, n_points)
        high_price = np.maximum(open_price, close_price) + rng.uniform(0, 0.001, n_points)
        low_price = np.minimum(open_price, close_price) - rng.uniform(0, 0.001, n_points)

        # Volume
        volume = rng.integers(100, 1000, n_points)

        data = pd.DataFrame({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        }, index=index)

        return data

    # 1. Generate and preprocess data
    data = generate_synthetic_data(days=250)
    data = preprocess_data(data)

    # 2. Initialize Backtest
    bt = Backtest(data, AsiaRangeLiquiditySweepUkReversalStrategy,
                  cash=100_000, commission=.002,
                  trade_on_close=False)

    # 3. Optimize the strategy
    stats = bt.optimize(
        max_asia_range_pct=np.arange(0.5, 3.1, 0.5).tolist(),
        risk_pct=np.arange(0.5, 2.1, 0.5).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.max_asia_range_pct > 0 and p.risk_pct > 0
    )

    print("Best stats:\n", stats)
    print("Best parameters:\n", stats._strategy)

    # 4. Save results to JSON
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Handle cases where no trades were made
    win_rate = stats.get('Win Rate [%]', 0)
    sharpe = stats.get('Sharpe Ratio', 0)

    with open(os.path.join(results_dir, 'temp_result.json'), 'w') as f:
        json.dump({
            'strategy_name': 'asia_range_liquidity_sweep_uk_reversal',
            'return': stats.get('Return [%]', 0),
            'sharpe': sharpe if np.isfinite(sharpe) else 0,
            'max_drawdown': stats.get('Max. Drawdown [%]', 0),
            'win_rate': win_rate if np.isfinite(win_rate) else 0,
            'total_trades': stats.get('# Trades', 0)
        }, f, indent=2)

    # 5. Generate plot
    plot_filename = os.path.join(results_dir, 'asia_range_liquidity_sweep_uk_reversal.html')
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")
