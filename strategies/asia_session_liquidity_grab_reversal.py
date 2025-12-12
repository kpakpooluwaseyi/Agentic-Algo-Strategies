import pandas as pd
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from datetime import time

def generate_synthetic_data(days=200):
    """
    Generates synthetic 15-minute OHLC data simulating Asia session consolidation
    and London session liquidity grabs.
    """
    rng = np.random.default_rng(seed=42)
    n_periods = days * 24 * 4  # 15-min periods in a day
    index = pd.date_range(start='2023-01-01', periods=n_periods, freq='15min', tz='UTC')

    # Base price movement
    price = 1.0
    prices = [price]
    for _ in range(1, n_periods):
        price += rng.normal(0, 0.0005)
        prices.append(price)

    df = pd.DataFrame(index=index)
    df['Close'] = prices
    df['Open'] = df['Close'].shift(1)
    df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0.0001, 0.0005, size=n_periods)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0.0001, 0.0005, size=n_periods)
    df.iloc[0, df.columns.get_loc('Open')] = df.iloc[0, df.columns.get_loc('Close')] # Set first Open

    # Simulate session dynamics
    for day in range(days):
        day_start_idx = day * 96

        # Asia Session (00:00 - 08:00 UTC) - Low volatility
        asia_start_idx = day_start_idx
        asia_end_idx = day_start_idx + 32
        asia_range = df.iloc[asia_start_idx:asia_end_idx]
        asia_mid_price = (asia_range['High'].max() + asia_range['Low'].min()) / 2
        df.loc[asia_range.index, ['Open', 'High', 'Low', 'Close']] *= 0.1 # Reduce vol
        df.loc[asia_range.index, ['Open', 'High', 'Low', 'Close']] += asia_mid_price * 0.9

        # London Session (08:00 - 16:00 UTC) - Potential for liquidity grab
        london_start_idx = asia_end_idx

        # Decide if a grab will happen
        if rng.random() > 0.3: # 70% chance of a grab
            asia_high = df.iloc[asia_start_idx:asia_end_idx]['High'].max()
            asia_low = df.iloc[asia_start_idx:asia_end_idx]['Low'].min()

            grab_time_idx = rng.integers(london_start_idx + 1, london_start_idx + 8) # Grab in first 2 hours

            if rng.random() > 0.5: # Bearish grab
                spike_high = asia_high + rng.uniform(0.0005, 0.0015)
                df.at[df.index[grab_time_idx], 'High'] = spike_high
                df.at[df.index[grab_time_idx], 'Open'] = df.iloc[grab_time_idx - 1]['Close']
                df.at[df.index[grab_time_idx], 'Close'] = df.iloc[grab_time_idx - 1]['Open'] - 0.0001 # Engulfing
                df.at[df.index[grab_time_idx], 'Low'] = min(df.at[df.index[grab_time_idx], 'Close'], df.at[df.index[grab_time_idx], 'Open'])
            else: # Bullish grab
                spike_low = asia_low - rng.uniform(0.0005, 0.0015)
                df.at[df.index[grab_time_idx], 'Low'] = spike_low
                df.at[df.index[grab_time_idx], 'Open'] = df.iloc[grab_time_idx - 1]['Close']
                df.at[df.index[grab_time_idx], 'Close'] = df.iloc[grab_time_idx - 1]['Open'] + 0.0001 # Engulfing
                df.at[df.index[grab_time_idx], 'High'] = max(df.at[df.index[grab_time_idx], 'Close'], df.at[df.index[grab_time_idx], 'Open'])

    df = df.dropna()
    return df

def preprocess_data(df):
    """
    Identifies trading sessions and calculates daily Asia session High/Low.
    """
    df['date'] = df.index.date

    # Define session times
    asia_session_start = time(0, 0)
    asia_session_end = time(8, 0)

    df['is_asia'] = (df.index.time >= asia_session_start) & (df.index.time < asia_session_end)

    asia_highs = df[df['is_asia']].groupby('date')['High'].max()
    asia_lows = df[df['is_asia']].groupby('date')['Low'].min()

    df['HOA'] = df['date'].map(asia_highs).ffill()
    df['LOA'] = df['date'].map(asia_lows).ffill()

    df['asia_range_pct'] = ((df['HOA'] - df['LOA']) / df['LOA']) * 100

    df = df.dropna()
    return df

class AsiaSessionLiquidityGrabReversalStrategy(Strategy):
    max_asia_range_pct = 2.0  # Max Asia range in percent
    sl_buffer_pct = 0.1 # Stop loss buffer in percent
    risk_pct = 1.0 # Percentage of equity to risk per trade

    def init(self):
        self.uk_session_start = time(8, 0)
        self.session_end = time(16, 0)
        self.daily_trade_taken = False
        self.current_day = None

    def next(self):
        current_time = self.data.index[-1].time()
        current_date = self.data.index[-1].date()

        if self.current_day != current_date:
            self.current_day = current_date
            self.daily_trade_taken = False

        if current_time < self.uk_session_start or current_time >= self.session_end or self.daily_trade_taken:
            return

        hoa = self.data.HOA[-1]
        loa = self.data.LOA[-1]
        asia_range = self.data.asia_range_pct[-1]

        if asia_range > self.max_asia_range_pct or self.position:
            return

        # Short Entry Logic
        if self.data.High[-1] > hoa:
            is_bearish_engulfing = (self.data.Close[-1] < self.data.Open[-1] and
                                   self.data.Open[-1] >= self.data.Close[-2] and
                                   self.data.Close[-1] <= self.data.Open[-2] and
                                   (self.data.Open[-1] - self.data.Close[-1]) > (self.data.Close[-2] - self.data.Open[-2]))

            if is_bearish_engulfing:
                sl_price = self.data.High[-1] * (1 + self.sl_buffer_pct / 100)
                tp_price = loa
                entry_price = self.data.Close[-1]

                if tp_price < entry_price: # TP must be below entry for a short
                    sl_distance_points = abs(entry_price - sl_price)
                    if sl_distance_points == 0: return

                    risk_per_trade = self.equity * (self.risk_pct / 100)
                    size = int(risk_per_trade / sl_distance_points)

                    if size > 0 and self.equity > size * entry_price:
                        self.sell(size=size, sl=sl_price, tp=tp_price)
                        self.daily_trade_taken = True

        # Long Entry Logic
        elif self.data.Low[-1] < loa:
            is_bullish_engulfing = (self.data.Close[-1] > self.data.Open[-1] and
                                    self.data.Close[-1] >= self.data.Open[-2] and
                                    self.data.Open[-1] <= self.data.Close[-2] and
                                    (self.data.Close[-1] - self.data.Open[-1]) > (self.data.Open[-2] - self.data.Close[-2]))

            if is_bullish_engulfing:
                sl_price = self.data.Low[-1] * (1 - self.sl_buffer_pct / 100)
                tp_price = hoa
                entry_price = self.data.Close[-1]

                if tp_price > entry_price: # TP must be above entry for a long
                    sl_distance_points = abs(entry_price - sl_price)
                    if sl_distance_points == 0: return

                    risk_per_trade = self.equity * (self.risk_pct / 100)
                    size = int(risk_per_trade / sl_distance_points)

                    if size > 0 and self.equity > size * entry_price:
                        self.buy(size=size, sl=sl_price, tp=tp_price)
                        self.daily_trade_taken = True

if __name__ == '__main__':
    # Generate and preprocess data
    data = generate_synthetic_data(days=250)
    data = preprocess_data(data)

    # Initialize Backtest
    bt = Backtest(data, AsiaSessionLiquidityGrabReversalStrategy, cash=100_000, commission=.002)

    # Optimize
    stats = bt.optimize(
        max_asia_range_pct=np.arange(0.5, 3.1, 0.5).tolist(),
        sl_buffer_pct=np.arange(0.05, 0.31, 0.05).tolist(),
        risk_pct=np.arange(0.5, 2.1, 0.5).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.max_asia_range_pct > 0
    )

    print("Best stats:\n", stats)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Save results
    win_rate = stats.get('Win Rate [%]', 0)
    sharpe = stats.get('Sharpe Ratio', 0)

    if np.isnan(sharpe):
        sharpe = None

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'asia_session_liquidity_grab_reversal',
            'return': stats.get('Return [%]', 0),
            'sharpe': sharpe,
            'max_drawdown': stats.get('Max. Drawdown [%]', 0),
            'win_rate': win_rate,
            'total_trades': stats.get('# Trades', 0)
        }, f, indent=2)

    # Generate plot
    bt.plot(filename='results/asia_session_liquidity_grab_reversal.html')
