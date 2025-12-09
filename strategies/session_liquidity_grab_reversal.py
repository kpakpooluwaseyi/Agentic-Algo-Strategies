import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import os

def is_bearish_engulfing(candle1, candle2):
    """Checks for a bearish engulfing pattern."""
    return candle1['Open'] < candle1['Close'] and candle2['Open'] > candle2['Close'] and \
           candle2['Open'] > candle1['Close'] and candle2['Close'] < candle1['Open']

def is_bullish_engulfing(candle1, candle2):
    """Checks for a bullish engulfing pattern."""
    return candle1['Open'] > candle1['Close'] and candle2['Open'] < candle2['Close'] and \
           candle2['Open'] < candle1['Close'] and candle2['Close'] > candle1['Open']

def is_bullish_hammer(candle):
    """Checks for a bullish hammer pattern."""
    body = abs(candle['Close'] - candle['Open'])
    lower_wick = candle['Open'] - candle['Low'] if candle['Close'] > candle['Open'] else candle['Close'] - candle['Low']
    upper_wick = candle['High'] - candle['Close'] if candle['Close'] > candle['Open'] else candle['High'] - candle['Open']
    return lower_wick > body * 2 and upper_wick < body

def generate_synthetic_data():
    """Generates synthetic 15-minute data with Asia session liquidity grabs."""
    # Create a 15-minute date range
    index = pd.date_range(start='2023-01-01 00:00', end='2023-03-31 23:45', freq='15min')
    n = len(index)
    data = pd.DataFrame(index=index)

    # Base price movement
    price = 100 + np.random.randn(n).cumsum() * 0.1
    data['Open'] = price
    data['High'] = price + np.random.uniform(0, 0.5, n)
    data['Low'] = price - np.random.uniform(0, 0.5, n)
    data['Close'] = data['Open'] + np.random.uniform(-0.3, 0.3, n)

    # Ensure OHLC consistency
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    # --- Inject specific patterns ---
    # Find UK session start times
    uk_session_starts = data.index[(data.index.hour == 8) & (data.index.minute == 0)]

    for i in range(min(5, len(uk_session_starts) -1)): # Inject 5 patterns
        start_idx = data.index.get_loc(uk_session_starts[i])

        # Define Asia range (00:00 - 08:00 UTC)
        asia_range = data.iloc[start_idx-32:start_idx]
        asia_high = asia_range['High'].max()
        asia_low = asia_range['Low'].min()

        if i % 2 == 0: # Create a bearish grab
            # Grab liquidity above Asia High
            grab_idx = start_idx + 2
            data.loc[data.index[grab_idx], 'High'] = asia_high + 0.2
            data.loc[data.index[grab_idx], 'Open'] = asia_high + 0.1
            data.loc[data.index[grab_idx], 'Close'] = asia_high - 0.1

            # Reversal candle (bearish engulfing)
            reversal_idx = grab_idx + 1
            data.loc[data.index[reversal_idx], 'Open'] = data.loc[data.index[grab_idx], 'Close'] + 0.05
            data.loc[data.index[reversal_idx], 'Close'] = data.loc[data.index[reversal_idx], 'Open'] - 0.3
            data.loc[data.index[reversal_idx], 'High'] = data.loc[data.index[reversal_idx], 'Open']
            data.loc[data.index[reversal_idx], 'Low'] = data.loc[data.index[reversal_idx], 'Close'] - 0.1


        else: # Create a bullish grab
            # Grab liquidity below Asia Low
            grab_idx = start_idx + 2
            data.loc[data.index[grab_idx], 'Low'] = asia_low - 0.2
            data.loc[data.index[grab_idx], 'Open'] = asia_low - 0.1
            data.loc[data.index[grab_idx], 'Close'] = asia_low + 0.1

            # Reversal candle (bullish engulfing)
            reversal_idx = grab_idx + 1
            data.loc[data.index[reversal_idx], 'Open'] = data.loc[data.index[grab_idx], 'Close'] - 0.05
            data.loc[data.index[reversal_idx], 'Close'] = data.loc[data.index[reversal_idx], 'Open'] + 0.3
            data.loc[data.index[reversal_idx], 'Low'] = data.loc[data.index[reversal_idx], 'Open']
            data.loc[data.index[reversal_idx], 'High'] = data.loc[data.index[reversal_idx], 'Close'] + 0.1


    return data


def preprocess_data(df, asia_start_hour=0, asia_end_hour=8):
    """Calculates session ranges and adds them to the DataFrame."""
    df['hour'] = df.index.hour

    # Identify sessions
    df['is_asia'] = (df['hour'] >= asia_start_hour) & (df['hour'] < asia_end_hour)
    df['is_uk'] = (df['hour'] >= 8) & (df['hour'] < 16)

    # Calculate daily Asia High/Low
    asia_session_data = df[df['is_asia']].copy()
    daily_asia_high = asia_session_data.groupby(asia_session_data.index.date)['High'].max()
    daily_asia_low = asia_session_data.groupby(asia_session_data.index.date)['Low'].min()

    # Calculate previous day's 50% level
    daily_high = df.groupby(df.index.date)['High'].max().shift(1)
    daily_low = df.groupby(df.index.date)['Low'].min().shift(1)
    daily_mid = (daily_high + daily_low) / 2

    # Map to the whole day
    df['asia_high'] = pd.Series(df.index.date, index=df.index).map(daily_asia_high)
    df['asia_low'] = pd.Series(df.index.date, index=df.index).map(daily_asia_low)
    df['prev_day_mid'] = pd.Series(df.index.date, index=df.index).map(daily_mid)

    df.ffill(inplace=True)
    return df

class SessionLiquidityGrabReversalStrategy(Strategy):
    asia_range_max_pct = 2.0

    def init(self):
        self.current_day = None
        self.asia_high = None
        self.asia_low = None
        self.liquidity_grabbed = None # 'high' or 'low'
        self.tp1_hit = False

    def next(self):
        # --- TP/Exit Logic for open positions ---
        if self.position:
            # Short position TP management
            if self.position.is_short:
                if not self.tp1_hit and self.data.Low[-1] <= self.asia_low:
                    self.position.close(portion=0.5)
                    self.tp1_hit = True
                if self.tp1_hit and self.data.Low[-1] <= self.data.prev_day_mid[-1]:
                    self.position.close()

            # Long position TP management
            else:
                if not self.tp1_hit and self.data.High[-1] >= self.asia_high:
                    self.position.close(portion=0.5)
                    self.tp1_hit = True
                if self.tp1_hit and self.data.High[-1] >= self.data.prev_day_mid[-1]:
                    self.position.close()

        # Daily state reset
        today = self.data.index[-1].date()
        if self.current_day != today:
            self.current_day = today
            self.asia_high = self.data.asia_high[-1]
            self.asia_low = self.data.asia_low[-1]
            self.liquidity_grabbed = None
            self.tp1_hit = False

            # Check Asia range prerequisite
            if self.asia_high and self.asia_low:
                asia_range = (self.asia_high - self.asia_low) / self.asia_low * 100
                if asia_range > self.asia_range_max_pct:
                    self.asia_high = None # Invalidate today's trading

        # Only trade during UK session and if Asia range is valid
        if not self.data.is_uk[-1] or not self.asia_high or self.position:
            return

        # --- Entry Logic ---

        # 1. Detect Liquidity Grab
        if not self.liquidity_grabbed:
            if self.data.High[-1] > self.asia_high:
                self.liquidity_grabbed = 'high'
            elif self.data.Low[-1] < self.asia_low:
                self.liquidity_grabbed = 'low'

        # 2. Look for Reversal Confirmation
        if self.liquidity_grabbed and len(self.data) > 1 and not self.position:
            if self.liquidity_grabbed == 'high':
                # Bearish engulfing after grab
                candle1 = {'Open': self.data.Open[-2], 'Close': self.data.Close[-2]}
                candle2 = {'Open': self.data.Open[-1], 'Close': self.data.Close[-1]}
                if is_bearish_engulfing(candle1, candle2):
                    sl = self.data.High[-2] + (self.data.High[-2] * 0.001)
                    # Pre-validate SL and TP
                    if sl > self.data.Close[-1] and self.asia_low < self.data.Close[-1]:
                        self.sell(sl=sl)
                        self.liquidity_grabbed = 'done'

            elif self.liquidity_grabbed == 'low':
                 # Bullish engulfing or Hammer after grab
                candle1 = {'Open': self.data.Open[-2], 'Close': self.data.Close[-2]}
                candle2 = {'Open': self.data.Open[-1], 'Close': self.data.Close[-1]}
                hammer_candle = {'Open': self.data.Open[-1], 'Close': self.data.Close[-1], 'High': self.data.High[-1], 'Low': self.data.Low[-1]}
                if is_bullish_engulfing(candle1, candle2) or is_bullish_hammer(hammer_candle):
                    sl = self.data.Low[-2] - (self.data.Low[-2] * 0.001)
                    # Pre-validate SL and TP
                    if sl < self.data.Close[-1] and self.asia_high > self.data.Close[-1]:
                        self.buy(sl=sl)
                        self.liquidity_grabbed = 'done'


if __name__ == '__main__':
    data = generate_synthetic_data()
    data = preprocess_data(data)

    bt = Backtest(data, SessionLiquidityGrabReversalStrategy, cash=100_000, commission=.002)

    stats = bt.optimize(
        asia_range_max_pct=np.arange(1.0, 3.5, 0.5).tolist(),
        maximize='Sharpe Ratio'
    )

    print(stats)

    os.makedirs('results', exist_ok=True)

    # Check for trades before saving results
    if stats['# Trades'] > 0:
        win_rate = float(stats['Win Rate [%]'])
        sharpe = float(stats['Sharpe Ratio'])
    else:
        win_rate = 0.0
        sharpe = 0.0

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'session_liquidity_grab_reversal',
            'return': float(stats['Return [%]']),
            'sharpe': sharpe,
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': win_rate,
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    # Use the best strategy instance from optimization for plotting
    if stats._strategy:
        bt.plot(filename="results/session_liquidity_grab_reversal.html")
    else:
        print("No best strategy found to plot.")
