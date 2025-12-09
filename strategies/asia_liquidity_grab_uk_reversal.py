
import json
import os

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy


def generate_forex_data(days=200):
    """
    Generates synthetic 24-hour forex data with distinct session-based movements.
    """
    rng = np.random.default_rng(seed=42)
    n_points = days * 24 * 4  # 15-min intervals
    index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_points, freq='15min', tz='UTC'))

    # Base price movement
    price = 1.1000
    base_returns = rng.normal(loc=0, scale=0.0001, size=n_points)

    # Session-based volatility
    time = index.hour
    asia_session = (time >= 0) & (time < 8)
    london_session = (time >= 8) & (time < 16)
    ny_session = (time >= 13) & (time < 22)

    volatility = np.ones(n_points)
    volatility[asia_session] = rng.uniform(0.5, 0.8, size=asia_session.sum())
    volatility[london_session] = rng.uniform(1.2, 1.8, size=london_session.sum())
    volatility[ny_session] = rng.uniform(1.5, 2.2, size=ny_session.sum())

    returns = base_returns * volatility
    price_path = price * np.exp(np.cumsum(returns))

    # Create ohlc
    ohlc = pd.DataFrame(index=index)
    ohlc['Open'] = pd.Series(price_path).resample('15min').first()
    ohlc['High'] = pd.Series(price_path).resample('15min').max()
    ohlc['Low'] = pd.Series(price_path).resample('15min').min()
    ohlc['Close'] = pd.Series(price_path).resample('15min').last()

    # Generate some textbook patterns
    for _ in range(int(days / 10)):
        day = rng.integers(1, days - 2)
        start_idx = day * 96
        # Asia range
        asia_start, asia_end = start_idx, start_idx + 32
        asia_low = ohlc['Low'][asia_start:asia_end].min()
        asia_high = ohlc['High'][asia_start:asia_end].max()

        # London liquidity grab
        london_start, london_end = start_idx + 32, start_idx + 64
        grab_idx = london_start + rng.integers(4, 12)

        if grab_idx + 2 < len(ohlc):
            # Spike above Asia high
            ohlc.loc[ohlc.index[grab_idx], 'High'] = asia_high * 1.001
            ohlc.loc[ohlc.index[grab_idx], 'Open'] = ohlc.loc[ohlc.index[grab_idx-1], 'Close']
            ohlc.loc[ohlc.index[grab_idx], 'Close'] = ohlc.loc[ohlc.index[grab_idx], 'High'] * 0.9995

            # Bearish engulfing
            ohlc.loc[ohlc.index[grab_idx+1], 'Open'] = ohlc.loc[ohlc.index[grab_idx], 'Close'] * 1.0002
            ohlc.loc[ohlc.index[grab_idx+1], 'Close'] = ohlc.loc[ohlc.index[grab_idx-1], 'Open'] * 0.999
            ohlc.loc[ohlc.index[grab_idx+1], 'High'] = ohlc.loc[ohlc.index[grab_idx+1], 'Open'] * 1.0001
            ohlc.loc[ohlc.index[grab_idx+1], 'Low'] = ohlc.loc[ohlc.index[grab_idx+1], 'Close'] * 0.9999

            # Trend down to Asia low
            reversal_end = grab_idx + 1 + rng.integers(10, 20)
            if reversal_end < len(ohlc):
                price_decline = np.linspace(ohlc.loc[ohlc.index[grab_idx+1], 'Close'], asia_low * 0.9998, reversal_end - (grab_idx + 1))
                ohlc.loc[ohlc.index[grab_idx+2:reversal_end+1], 'Close'] = price_decline
                ohlc.loc[ohlc.index[grab_idx+2:reversal_end+1], 'Open'] = price_decline * 1.0001


    ohlc.ffill(inplace=True)
    return ohlc.dropna()

def preprocess_data(df):
    """
    Adds session-based indicators and other necessary columns to the DataFrame.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)

    # Identify sessions
    df['date'] = df.index.date
    df['time'] = df.index.time

    df['session_id'] = (df['date'] != df['date'].shift(1)).cumsum()

    # Asia Session
    is_asia = (df['time'] >= pd.to_datetime('00:00').time()) & (df['time'] <= pd.to_datetime('08:00').time())
    asia_high = df[is_asia].groupby('session_id')['High'].transform('max')
    asia_low = df[is_asia].groupby('session_id')['Low'].transform('min')
    df['asia_high'] = asia_high.ffill()
    df['asia_low'] = asia_low.ffill()

    # London Session
    is_london = (df['time'] >= pd.to_datetime('08:00').time()) & (df['time'] <= pd.to_datetime('17:00').time())
    df['is_london'] = is_london

    # US Session Open Avoidance
    is_us_open = (df['time'] >= pd.to_datetime('13:00').time()) & (df['time'] <= pd.to_datetime('14:00').time())
    df['is_us_open'] = is_us_open

    # Asia Range Filter
    df['asia_range'] = (df['asia_high'] - df['asia_low']) / df['asia_low']
    df['asia_range_ok'] = df['asia_range'] < 0.02

    # Previous Day's 50% Level
    daily_high = df.groupby('session_id')['High'].transform('max')
    daily_low = df.groupby('session_id')['Low'].transform('min')
    prev_daily_high = daily_high.shift(96) # Shift by one day (96 * 15min)
    prev_daily_low = daily_low.shift(96)
    df['pd_50'] = (prev_daily_high + prev_daily_low) / 2

    # Bearish Engulfing Pattern
    body_high = df[['Open', 'Close']].max(axis=1)
    body_low = df[['Open', 'Close']].min(axis=1)
    is_red = df['Close'] < df['Open']
    is_green = df['Close'] > df['Open']

    prev_body_high = body_high.shift(1)
    prev_body_low = body_low.shift(1)
    prev_is_green = is_green.shift(1)

    df['is_bearish_engulfing'] = (
        is_red &
        prev_is_green &
        (body_high > prev_body_high) &
        (body_low < prev_body_low)
    )

    df.dropna(inplace=True)
    return df


class AsiaLiquidityGrabUkReversalStrategy(Strategy):
    sl_buffer_pct = 0.1

    def init(self):
        # Pre-calculated in preprocess_data
        self.asia_high = self.data.df['asia_high']
        self.asia_low = self.data.df['asia_low']
        self.is_london = self.data.df['is_london']
        self.is_us_open = self.data.df['is_us_open']
        self.asia_range_ok = self.data.df['asia_range_ok']
        self.pd_50 = self.data.df['pd_50']
        self.is_bearish_engulfing = self.data.df['is_bearish_engulfing']

    def next(self):
        # Ensure we have enough data for the lookback
        if len(self.data) < 2:
            return

        # Conditions for trading
        can_trade = (
            self.is_london[self.i] and
            not self.is_us_open[self.i] and
            self.asia_range_ok[self.i] and
            not self.position
        )

        if not can_trade:
            return

        # Entry Logic: A two-candle pattern
        # Candle[-2] is the liquidity grab
        # Candle[-1] is the bearish engulfing confirmation

        liquidity_grab_candle_high = self.data.High[-2]
        confirmation_candle_is_engulfing = self.is_bearish_engulfing[-1]

        if confirmation_candle_is_engulfing and (liquidity_grab_candle_high > self.asia_high[-2]):
            # Entry: Next candle open
            sl = liquidity_grab_candle_high * (1 + self.sl_buffer_pct/100)
            tp1 = self.asia_low[-1]
            tp2 = self.pd_50[-1]

            # Risk management: check for valid TP levels
            if self.data.Close[-1] > tp1 and self.data.Close[-1] > tp2:
                # Sell half for TP1, half for TP2
                self.sell(size=0.5, sl=sl, tp=tp1)
                self.sell(size=0.5, sl=sl, tp=tp2)

if __name__ == '__main__':
    data = generate_forex_data(days=500)
    data = preprocess_data(data)

    bt = Backtest(data, AsiaLiquidityGrabUkReversalStrategy, cash=100000, commission=.0002)

    stats = bt.optimize(
        sl_buffer_pct=np.arange(0.1, 1.0, 0.1).tolist(),
        maximize='Sharpe Ratio'
    )

    print(stats)

    os.makedirs('results', exist_ok=True)

    # Extract stats from the best run
    best_stats = stats

    # Handle cases with no trades
    if best_stats and best_stats.get('# Trades', 0) > 0:
        win_rate = best_stats['Win Rate [%]']
        sharpe = best_stats.get('Sharpe Ratio', 0)
        total_trades = best_stats['# Trades']
        return_pct = best_stats['Return [%]']
        max_drawdown = best_stats['Max. Drawdown [%]']
    else:
        win_rate = 0
        sharpe = 0
        total_trades = 0
        return_pct = 0
        max_drawdown = 0

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'asia_liquidity_grab_uk_reversal',
            'return': return_pct,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }, f, indent=2)

    bt.plot(filename='results/plot.html', open_browser=False)
