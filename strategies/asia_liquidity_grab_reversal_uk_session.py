from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
from enum import Enum

# --- State Machine ---
class State(Enum):
    WAITING = 0
    GRAB_HIGH_DETECTED = 1
    GRAB_LOW_DETECTED = 2

# --- Candlestick Pattern Recognition ---
def is_bullish_engulfing(df, i):
    """Checks for a bullish engulfing pattern at index i."""
    if i < 1:
        return False
    current = df.iloc[i]
    previous = df.iloc[i - 1]

    # Previous candle must be bearish, current must be bullish.
    if not (previous['Close'] < previous['Open'] and current['Close'] > current['Open']):
        return False
    # Current candle's body must engulf the previous candle's body.
    if not (current['Open'] <= previous['Close'] and current['Close'] >= previous['Open']):
        return False
    return True

def is_bearish_engulfing(df, i):
    """Checks for a bearish engulfing pattern at index i."""
    if i < 1:
        return False
    current = df.iloc[i]
    previous = df.iloc[i - 1]

    # Previous candle must be bullish, current must be bearish.
    if not (previous['Close'] > previous['Open'] and current['Close'] < current['Open']):
        return False
    # Current candle's body must engulf the previous candle's body.
    if not (current['Open'] >= previous['Close'] and current['Close'] <= previous['Open']):
        return False
    return True

# --- Data Preprocessing ---
def preprocess_data(df):
    """Adds session information and Asia session H/L to the DataFrame."""
    df.index = pd.to_datetime(df.index)

    df['hour'] = df.index.hour
    df['is_asia'] = (df['hour'] >= 0) & (df['hour'] < 8)
    df['is_uk'] = (df['hour'] >= 8) & (df['hour'] < 16)

    # Create a unique ID for each day to group sessions
    df['date'] = df.index.date
    df['session_id'] = (df['date'] != df['date'].shift()).cumsum()

    # Calculate Asia session High and Low and map them back
    asia_high_map = df[df['is_asia']].groupby('session_id')['High'].max()
    asia_low_map = df[df['is_asia']].groupby('session_id')['Low'].min()

    df['asia_high'] = df['session_id'].map(asia_high_map)
    df['asia_low'] = df['session_id'].map(asia_low_map)

    # Forward-fill the Asia H/L to make them available to the UK session of the same day
    df['asia_high'] = df.groupby('session_id')['asia_high'].ffill()
    df['asia_low'] = df.groupby('session_id')['asia_low'].ffill()

    df['asia_range'] = df['asia_high'] - df['asia_low']
    df['asia_range_pct'] = (df['asia_range'] / df['asia_low']) * 100

    # Clean up and drop rows where Asia H/L is not available (e.g., first day)
    df.drop(columns=['hour', 'date', 'session_id'], inplace=True)
    df.dropna(inplace=True)

    return df

class AsiaLiquidityGrabReversalUkSessionStrategy(Strategy):
    asia_range_pct_max = 2.0

    def init(self):
        self.state = State.WAITING
        # Access data directly from self.data to avoid stale cached data during optimization

    def next(self):
        # Use direct access to DataFrame columns for checks
        df = self.data.df
        is_uk = df['is_uk'].iloc[-1]
        asia_range_pct = df['asia_range_pct'].iloc[-1]
        asia_high = df['asia_high'].iloc[-1]
        asia_low = df['asia_low'].iloc[-1]

        # If a new day starts, reset the state machine.
        if self.data.index[-1].hour == 0 and self.data.index[-1].minute == 0:
            self.state = State.WAITING

        # Pre-requisite checks
        if self.position or not is_uk or asia_range_pct > self.asia_range_pct_max:
            return

        # --- State Machine Logic ---
        if self.state == State.WAITING:
            if self.data.High[-1] > asia_high:
                self.state = State.GRAB_HIGH_DETECTED
            elif self.data.Low[-1] < asia_low:
                self.state = State.GRAB_LOW_DETECTED

        elif self.state == State.GRAB_HIGH_DETECTED:
            if is_bearish_engulfing(df, len(self.data)-1):
                sl = self.data.High[-1] * 1.001
                tp = asia_low
                if tp < self.data.Close[-1]:
                    self.sell(sl=sl, tp=tp)
                self.state = State.WAITING
            elif self.data.Close[-1] < asia_high:
                self.state = State.WAITING

        elif self.state == State.GRAB_LOW_DETECTED:
            if is_bullish_engulfing(df, len(self.data)-1):
                sl = self.data.Low[-1] * 0.999
                tp = asia_high
                if tp > self.data.Close[-1]:
                    self.buy(sl=sl, tp=tp)
                self.state = State.WAITING
            elif self.data.Close[-1] > asia_low:
                self.state = State.WAITING

if __name__ == '__main__':
    def generate_synthetic_data(days=200):
        n = days * 24 * 4  # 15-min intervals
        dates = pd.date_range(start='2023-01-01', periods=n, freq='15min')
        price = 1.1000 + np.random.randn(n).cumsum() * 0.0001 + np.sin(np.arange(n) / (24 * 4 * 7)) * 0.005
        df = pd.DataFrame(index=dates)
        df['Open'] = price
        df['High'] = df['Open'] + np.random.uniform(0.0001, 0.0005, n)
        df['Low'] = df['Open'] - np.random.uniform(0.0001, 0.0005, n)
        df['Close'] = df['Open'] + np.random.uniform(-0.0003, 0.0003, n)
        df['High'] = np.maximum(df['High'], df[['Open', 'Close']].max(axis=1))
        df['Low'] = np.minimum(df['Low'], df[['Open', 'Close']].min(axis=1))
        df['Volume'] = np.random.randint(100, 1000, n)
        return df

    data = generate_synthetic_data(days=100)
    processed_data = preprocess_data(data)

    bt = Backtest(processed_data, AsiaLiquidityGrabReversalUkSessionStrategy, cash=100_000, commission=.0002)

    stats = bt.optimize(
        asia_range_pct_max=np.arange(1.0, 3.1, 0.5).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.asia_range_pct_max > 0
    )

    import os
    os.makedirs('results', exist_ok=True)

    results_dict = {
        'strategy_name': 'asia_liquidity_grab_reversal_uk_session',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    for key, value in results_dict.items():
        if isinstance(value, (np.floating, float)) and np.isnan(value):
            results_dict[key] = None
        elif isinstance(value, np.integer):
            results_dict[key] = int(value)
        elif isinstance(value, np.floating):
             results_dict[key] = float(value)

    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(stats)

    try:
        if stats.get('# Trades', 0) > 0:
            bt.plot(filename='results/asia_liquidity_grab_reversal_uk_session.html', open_browser=False)
        else:
            print("No trades were made, skipping plot generation.")
    except Exception as e:
        print(f"Could not generate plot: {e}")
