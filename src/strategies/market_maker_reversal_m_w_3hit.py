from backtesting import Backtest, Strategy
from backtesting.lib import crossover, resample_apply
from backtesting.test import SMA
import pandas as pd
import numpy as np
import json

# --- Candlestick Pattern Helpers ---
def is_hammer(ohlc):
    """Detects a Hammer candlestick pattern."""
    open_price, high, low, close = ohlc.Open, ohlc.High, ohlc.Low, ohlc.Close
    body_size = abs(close - open_price)
    if body_size == 0: return False
    lower_wick = (min(open_price, close)) - low
    upper_wick = high - (max(open_price, close))
    return lower_wick > 2 * body_size and upper_wick < body_size

def is_inverted_hammer(ohlc):
    """Detects an Inverted Hammer candlestick pattern."""
    open_price, high, low, close = ohlc.Open, ohlc.High, ohlc.Low, ohlc.Close
    body_size = abs(close - open_price)
    if body_size == 0: return False
    upper_wick = high - (max(open_price, close))
    lower_wick = (min(open_price, close)) - low
    return upper_wick > 2 * body_size and lower_wick < body_size

def is_tweezer_tops(ohlc_prev, ohlc_curr):
    """Detects a Tweezer Tops candlestick pattern."""
    if not np.isclose(ohlc_prev.High, ohlc_curr.High, rtol=0.001):
        return False
    is_first_bullish = ohlc_prev.Close > ohlc_prev.Open
    is_second_bearish = ohlc_curr.Close < ohlc_curr.Open
    return is_first_bullish and is_second_bearish

class MarketMakerReversalMW3HitStrategy(Strategy):
    ema50_period = 50
    ema200_period = 200

    def init(self):
        self.ema50_1h = resample_apply('1H', lambda x: pd.Series(x).ewm(span=self.ema50_period, adjust=False).mean(), self.data.Close)
        self.ema200_1h = resample_apply('1H', lambda x: pd.Series(x).ewm(span=self.ema200_period, adjust=False).mean(), self.data.Close)

        self.asia_hod = None
        self.asia_lod = None
        self.hod_touches = 0
        self.lod_touches = 0
        self.last_day = None
        self.w_trough1 = None
        self.m_peak1 = None
        self.hod_touch_registered = False
        self.lod_touch_registered = False

    def next(self):
        # Timezone conversion and daily reset
        current_time = self.data.index[-1]
        try:
            current_time_ny = current_time.tz_convert('America/New_York')
        except TypeError:
            current_time_ny = current_time.tz_localize('UTC').tz_convert('America/New_York')

        if self.last_day != current_time_ny.date():
            self.last_day = current_time_ny.date()
            self.asia_hod = self.asia_lod = None
            self.hod_touches = self.lod_touches = 0
            self.w_trough1 = self.m_peak1 = None
            self.hod_touch_registered = False
            self.lod_touch_registered = False

            # Calculate Asia Session HOD/LOD
            start_asia = pd.Timestamp(f'{self.last_day} 20:30', tz='America/New_York') - pd.Timedelta(days=1)
            end_asia = pd.Timestamp(f'{self.last_day} 03:00', tz='America/New_York')

            asia_session_mask = (self.data.index.tz_convert('America/New_York') >= start_asia) & \
                                (self.data.index.tz_convert('America/New_York') < end_asia)

            asia_data = self.data.df[asia_session_mask]
            if not asia_data.empty:
                self.asia_hod = max(asia_data.Open.max(), asia_data.Close.max())
                self.asia_lod = min(asia_data.Open.min(), asia_data.Close.min())

        # Check trading window
        time_ny = current_time_ny.time()
        is_trading_session = (time_ny >= pd.to_datetime('20:30').time() or time_ny <= pd.to_datetime('10:00').time())
        if not is_trading_session or self.asia_hod is None:
            return

        # Exit logic
        if self.trades:
            # Friday stop hunt exit
            if current_time_ny.weekday() == 4:
                for trade in self.trades:
                    if trade.is_long and self.data.Close[-1] < self.data.Open[-1]:
                        trade.close()
                    elif trade.is_short and self.data.Close[-1] > self.data.Open[-1]:
                        trade.close()

        # --- SHORT ENTRY LOGIC ---
        if not self.position:
            if self.data.High[-1] >= self.asia_hod and not self.hod_touch_registered:
                self.hod_touches += 1
                self.hod_touch_registered = True
            elif self.data.High[-1] < self.asia_hod:
                self.hod_touch_registered = False

            if self.hod_touches >= 3:
                # M-Formation: Find first peak
                if self.m_peak1 is None and self.data.High[-2] > self.data.High[-1] and self.data.High[-2] > self.data.High[-3]:
                    self.m_peak1 = self.data.High[-2]
                # Find second peak and check for confirmation candle
                elif self.m_peak1 is not None:
                    is_second_peak = self.data.High[-2] > self.data.High[-1] and self.data.High[-2] > self.data.High[-3] and self.data.High[-2] <= self.m_peak1
                    if is_second_peak:
                        ohlc_prev = self.data.df.iloc[-3]
                        ohlc_curr = self.data.df.iloc[-2]
                        if is_tweezer_tops(ohlc_prev, ohlc_curr) or is_hammer(ohlc_curr):
                            sl = self.data.High[-2] * 1.001
                            current_price = self.data.Close[-1]
                            tp1 = self.ema50_1h[-1]
                            tp2 = self.ema200_1h[-1]
                            # Open two trades for two TPs if TPs are valid
                            if tp1 < current_price:
                                self.sell(sl=sl, tp=tp1, size=0.5)
                            if tp2 < current_price:
                                self.sell(sl=sl, tp=tp2, size=0.5)
                        self.m_peak1 = None # Reset after attempting trade

        # --- LONG ENTRY LOGIC ---
        if not self.position:
            if self.data.Low[-1] <= self.asia_lod and not self.lod_touch_registered:
                self.lod_touches += 1
                self.lod_touch_registered = True
            elif self.data.Low[-1] > self.asia_lod:
                self.lod_touch_registered = False

            if self.lod_touches >= 3:
                # W-Formation: Find first trough
                if self.w_trough1 is None and self.data.Low[-2] < self.data.Low[-1] and self.data.Low[-2] < self.data.Low[-3]:
                    self.w_trough1 = self.data.Low[-2]
                # Find second trough and check for confirmation candle
                elif self.w_trough1 is not None:
                    is_second_trough = self.data.Low[-2] < self.data.Low[-1] and self.data.Low[-2] < self.data.Low[-3] and self.data.Low[-2] >= self.w_trough1
                    if is_second_trough:
                        ohlc_curr = self.data.df.iloc[-2]
                        if is_hammer(ohlc_curr) or is_inverted_hammer(ohlc_curr):
                            sl = self.data.Low[-2] * 0.999
                            current_price = self.data.Close[-1]
                            tp1 = self.ema50_1h[-1]
                            tp2 = self.ema200_1h[-1]
                            # Open two trades for two TPs if TPs are valid
                            if tp1 > current_price:
                                self.buy(sl=sl, tp=tp1, size=0.5)
                            if tp2 > current_price:
                                self.buy(sl=sl, tp=tp2, size=0.5)
                        self.w_trough1 = None # Reset

def generate_forex_data(days=90):
    """Generates synthetic 24h OHLCV Forex data for backtesting."""
    np.random.seed(42)
    dt_range = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=days * 96, freq='15min'))
    price = 1.1000
    prices = []
    for _ in range(len(dt_range)):
        price += np.random.uniform(-0.0005, 0.0005)
        prices.append(price)

    df = pd.DataFrame(prices, index=dt_range, columns=['Close'])
    df['Open'] = df['Close'].shift(1)
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 0.0003, len(df))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 0.0003, len(df))
    df['Volume'] = np.random.randint(100, 1000, len(df))
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

if __name__ == '__main__':
    data = generate_forex_data(days=180)

    bt = Backtest(data, MarketMakerReversalMW3HitStrategy, cash=10000, commission=.002)

    stats = bt.optimize(
        ema50_period=range(40, 60, 5),
        ema200_period=range(180, 220, 10),
        maximize='Sharpe Ratio'
    )

    import os
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'market_maker_reversal_m_w_3hit',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    bt.plot()
