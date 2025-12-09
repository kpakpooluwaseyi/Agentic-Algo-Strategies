
import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np
import json
import os

# --- Data Generation and Preprocessing ---

def generate_synthetic_data(periods=4000):
    """
    Generates synthetic 15-minute Forex data to simulate session-based price action.
    """
    # Create a 15-minute date range
    rng = pd.date_range('2023-01-01', periods=periods, freq='15min', tz='UTC')
    df = pd.DataFrame(index=rng)

    # Generate random price movements
    random_walk = np.random.normal(loc=0.0001, scale=0.001, size=periods).cumsum()
    df['Open'] = 1.1000 + random_walk

    # Create OHLC data
    df['High'] = df['Open'] + np.random.uniform(0, 0.001, periods)
    df['Low'] = df['Open'] - np.random.uniform(0, 0.001, periods)
    df['Close'] = df['Open'] + np.random.normal(loc=0, scale=0.0005, size=periods)

    # Ensure High is the highest and Low is the lowest
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

    df['Volume'] = np.random.randint(100, 1000, size=periods)

    return df

def preprocess_data(df, asia_start='00:00', asia_end='08:00'):
    """
    Adds session information and candlestick patterns to the dataframe.
    """
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index)

    # Identify daily sessions
    df_copy['date'] = df_copy.index.date

    # Calculate Asia session High and Low
    asia_session = df_copy.between_time(asia_start, asia_end)
    daily_asia_range = asia_session.groupby(asia_session.index.date).agg(
        HOA=('High', 'max'),
        LOA=('Low', 'min')
    )

    # Map HOA and LOA to each row
    df_copy['HOA'] = df_copy['date'].map(daily_asia_range['HOA'])
    df_copy['LOA'] = df_copy['date'].map(daily_asia_range['LOA'])

    # Forward fill to make it available throughout the day
    df_copy['HOA'] = df_copy['HOA'].ffill()
    df_copy['LOA'] = df_copy['LOA'].ffill()

    # Asia Range filter
    df_copy['Asia_Range'] = df_copy['HOA'] - df_copy['LOA']
    df_copy['Asia_Range_Pct'] = (df_copy['Asia_Range'] / df_copy['Close']) * 100

    # Bearish Engulfing Pattern
    # Current candle is bearish, previous is bullish
    # Current body engulfs previous body
    body_high = df_copy[['Open', 'Close']].max(axis=1)
    body_low = df_copy[['Open', 'Close']].min(axis=1)

    is_bearish = df_copy['Close'] < df_copy['Open']
    is_bullish = df_copy['Close'] > df_copy['Open']

    engulfs_prev = (
        (body_high > body_high.shift(1)) &
        (body_low < body_low.shift(1))
    )

    df_copy['Bearish_Engulfing'] = (
        is_bearish &
        is_bullish.shift(1) &
        engulfs_prev
    )

    df_copy.dropna(inplace=True)

    return df_copy

# --- Strategy Definition ---

class AsiaLiquiditySweepReversalStrategy(Strategy):
    """
    A strategy that shorts reversals after a liquidity sweep above the Asia session high.
    """
    # Optimization parameters
    uk_start_hour = 8
    asia_range_max_pct = 2.0
    sl_buffer = 0.0005

    def init(self):
        # The pre-processed data is accessed directly via self.data
        # No indicators needed here as they are pre-calculated
        self.uk_session_open = False
        self.liquidity_sweep_high = None

    def next(self):
        current_time = self.data.index[-1].time()
        current_hour = self.data.index[-1].hour

        # Reset daily variables at midnight
        if current_hour == 0 and self.data.index[-1].minute == 0:
            self.uk_session_open = False
            self.liquidity_sweep_high = None

        # Check for UK session open
        if not self.uk_session_open and current_hour >= self.uk_start_hour:
            self.uk_session_open = True

        # --- Entry Conditions ---

        # Must be after UK open and before the end of the day to avoid overnight holds
        if self.uk_session_open and current_hour < 22:

            # 1. Price trades above HOA (liquidity sweep)
            # We track the highest point of the sweep
            if self.data.High[-1] > self.data.HOA[-1]:
                if self.liquidity_sweep_high is None or self.data.High[-1] > self.liquidity_sweep_high:
                    self.liquidity_sweep_high = self.data.High[-1]

            # Only proceed if a sweep has occurred today
            if self.liquidity_sweep_high is not None:

                # 2. Asia range is within the filter
                is_range_valid = self.data.Asia_Range_Pct[-1] < self.asia_range_max_pct

                # 3. Bearish engulfing pattern confirmation
                is_reversal_confirmed = self.data.Bearish_Engulfing[-1]

                # 4. Entry price must close BELOW HOA to confirm reversal
                is_below_hoa = self.data.Close[-1] < self.data.HOA[-1]

                # Execute short trade if all conditions are met
                if is_range_valid and is_reversal_confirmed and is_below_hoa and not self.position:

                    # Risk Management
                    stop_loss = self.liquidity_sweep_high + self.sl_buffer
                    take_profit = self.data.LOA[-1]

                    # Ensure TP is valid before placing trade (TP must be below current price for a short)
                    if take_profit < self.data.Close[-1]:
                        # Place the trade
                        self.sell(sl=stop_loss, tp=take_profit)


# --- Backtesting and Optimization ---

if __name__ == '__main__':
    # 1. Generate and preprocess data
    # Using 24h forex-like data is crucial for session-based strategies
    data = generate_synthetic_data(periods=8000) # ~3 months of 15m data
    processed_data = preprocess_data(data, asia_start='00:00', asia_end='08:00')

    # 2. Initialize backtest
    bt = Backtest(processed_data, AsiaLiquiditySweepReversalStrategy, cash=100000, commission=.0002)

    # 3. Optimize the strategy
    # Note: Narrowing ranges for faster execution.
    stats = bt.optimize(
        uk_start_hour=range(7, 10, 1),
        asia_range_max_pct=list(np.arange(0.5, 3.0, 0.5)),
        sl_buffer=list(np.arange(0.0001, 0.001, 0.0002)),
        maximize='Sharpe Ratio'
    )

    print("Best optimization stats:")
    print(stats)

    # 4. Save results to JSON
    os.makedirs('results', exist_ok=True)

    # Handle cases with no trades
    return_pct = stats.get('Return [%]', 0.0)
    sharpe_ratio = stats.get('Sharpe Ratio', 0.0)
    max_drawdown = stats.get('Max. Drawdown [%]', 0.0)
    win_rate = stats.get('Win Rate [%]', 0.0)
    total_trades = stats.get('# Trades', 0)

    # Ensure types are JSON serializable
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'asia_liquidity_sweep_reversal',
            'return': float(return_pct),
            'sharpe': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': int(total_trades)
        }, f, indent=2)

    print("\nResults saved to results/temp_result.json")

    # 5. Generate plot of the best run
    bt.plot()
