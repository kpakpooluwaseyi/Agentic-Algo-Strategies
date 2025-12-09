import json
import os
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

# --- Data Generation and Preprocessing ---

def generate_forex_data(days=90):
    """
    Generates synthetic 24-hour OHLCV data for a specified number of days.
    The data is generated at a 15-minute frequency.
    """
    date_rng = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=days * 24 * 4, freq='15min'))

    price_movements = np.random.randn(len(date_rng)) * 0.005
    close = 1.1000 + np.cumsum(price_movements)

    open_price = close - price_movements
    high = np.maximum(open_price, close) + np.random.uniform(0, 0.001, len(date_rng))
    low = np.minimum(open_price, close) - np.random.uniform(0, 0.001, len(date_rng))

    data = pd.DataFrame({'Open': open_price, 'High': high, 'Low': low, 'Close': close}, index=date_rng)
    return data

def get_session(hour):
    """Determines the trading session based on the UTC hour."""
    if 0 <= hour < 8: return 'Asia'
    elif 8 <= hour < 16: return 'UK'
    else: return 'US'

def preprocess_data(df):
    """
    Adds session information and Asia session high/low to the DataFrame.
    """
    df['Session'] = df.index.hour.map(get_session)

    asia_session_data = df[df['Session'] == 'Asia']
    daily_asia_high = asia_session_data.groupby(asia_session_data.index.date)['High'].max()
    daily_asia_low = asia_session_data.groupby(asia_session_data.index.date)['Low'].min()

    df['Asia_High'] = pd.Series(df.index.date, index=df.index).map(daily_asia_high).ffill()
    df['Asia_Low'] = pd.Series(df.index.date, index=df.index).map(daily_asia_low).ffill()

    df['Asia_Range'] = df['Asia_High'] - df['Asia_Low']
    df['Asia_Range_Pct'] = (df['Asia_Range'] / df['Asia_Low']) * 100

    df.dropna(inplace=True)
    return df

# --- Strategy Class ---

class AsiaLiquidityGrabReversalStrategy(Strategy):
    max_asia_range_pct = 2.0

    def init(self):
        self.traded_today_short = False
        self.traded_today_long = False
        self.current_day = None

    def _is_bearish_engulfing(self):
        if len(self.data.Close) < 2: return False
        c1_bullish = self.data.Close[-2] > self.data.Open[-2]
        c2_bearish = self.data.Close[-1] < self.data.Open[-1]
        engulfing = self.data.Open[-1] > self.data.Close[-2] and self.data.Close[-1] < self.data.Open[-2]
        return c1_bullish and c2_bearish and engulfing

    def _is_bullish_engulfing(self):
        if len(self.data.Close) < 2: return False
        c1_bearish = self.data.Close[-2] < self.data.Open[-2]
        c2_bullish = self.data.Close[-1] > self.data.Open[-1]
        engulfing = self.data.Open[-1] < self.data.Close[-2] and self.data.Close[-1] > self.data.Open[-2]
        return c1_bearish and c2_bullish and engulfing

    def next(self):
        today = self.data.index[-1].date()
        if self.current_day != today:
            self.current_day = today
            self.traded_today_short = False
            self.traded_today_long = False

        current_session = self.data.Session[-1]
        asia_high = self.data.Asia_High[-1]
        asia_low = self.data.Asia_Low[-1]
        asia_range_pct = self.data.Asia_Range_Pct[-1]

        if self.position: return
        if current_session not in ['UK', 'US']: return
        if asia_range_pct > self.max_asia_range_pct: return

        if not self.traded_today_short and self._is_bearish_engulfing():
            liquidity_grab_candle_high = self.data.High[-1]
            if liquidity_grab_candle_high > asia_high:
                self.sell(sl=liquidity_grab_candle_high, tp=asia_low)
                self.traded_today_short = True

        if not self.traded_today_long and self._is_bullish_engulfing():
            liquidity_grab_candle_low = self.data.Low[-1]
            if liquidity_grab_candle_low < asia_low:
                self.buy(sl=liquidity_grab_candle_low, tp=asia_high)
                self.traded_today_long = True

# --- Main Execution Block ---

if __name__ == '__main__':
    data = generate_forex_data(days=180)
    data = preprocess_data(data)

    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=10000, commission=.002)

    stats = bt.optimize(
        max_asia_range_pct=np.arange(0.5, 3.5, 0.5).tolist(),
        maximize='Sharpe Ratio'
    )

    print(stats)

    os.makedirs('results', exist_ok=True)

    # Handle cases with no trades
    if stats['# Trades'] > 0:
        win_rate = stats['Win Rate [%]']
        sharpe = stats['Sharpe Ratio']
    else:
        win_rate = 0
        sharpe = 0

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': float(stats['Return [%]']),
            'sharpe': float(sharpe),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(win_rate),
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    bt.plot()
