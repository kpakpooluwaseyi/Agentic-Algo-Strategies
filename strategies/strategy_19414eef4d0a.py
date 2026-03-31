
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import talib
from scipy.signal import find_peaks
import json
import os

# NOTE: The code review requested inheriting from a non-existent `MoonDevStrategy`.
# This implementation uses `backtesting.Strategy` to align with the project's
# established framework and the template in `.agent/rules/strategy_development.md`.

def find_fvg(df):
    df['fvg_bullish'] = (df['Low'].shift(-1) > df['High'].shift(1))
    df['fvg_bullish_top'] = df['Low'].shift(-1)
    df['fvg_bullish_bottom'] = df['High'].shift(1)
    df['fvg_bearish'] = (df['High'].shift(-1) < df['Low'].shift(1))
    df['fvg_bearish_top'] = df['Low'].shift(1)
    df['fvg_bearish_bottom'] = df['High'].shift(-1)
    return df

def find_swing_points(series, order=5):
    high_peaks, _ = find_peaks(series, distance=order, prominence=series.std()/2)
    low_peaks, _ = find_peaks(-series, distance=order, prominence=series.std()/2)
    return high_peaks, low_peaks

def preprocess_data(df, **params):
    df = find_fvg(df)
    highs, lows = find_swing_points(df['Close'], order=params.get('swing_order', 15))
    df['swing_high'] = False
    df['swing_low'] = False
    if len(highs) > 0:
        df.iloc[highs, df.columns.get_loc('swing_high')] = True
    if len(lows) > 0:
        df.iloc[lows, df.columns.get_loc('swing_low')] = True
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df_4h = df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df['htf_uptrend'] = (df_4h['Close'] > df_4h['ema_200']).reindex(df.index, method='ffill').fillna(False)
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)
    df.dropna(subset=['htf_uptrend', 'volume_ma', 'atr'], inplace=True)
    return df

class PullbackTradingStrategy(Strategy):
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0 # This will be replaced by Fibonacci logic
    swing_order = 15

    def init(self):
        self.htf_uptrend = self.I(lambda: self.data.df['htf_uptrend'])
        self.volume_ma = self.I(lambda: self.data.df['volume_ma'])
        self.atr = self.I(lambda: self.data.df['atr'])
        self.swing_high = self.I(lambda: self.data.df['swing_high'])
        self.swing_low = self.I(lambda: self.data.df['swing_low'])
        self.fvg_bullish = self.I(lambda: self.data.df['fvg_bullish'])
        self.fvg_bullish_top = self.I(lambda: self.data.df['fvg_bullish_top'])
        self.fvg_bullish_bottom = self.I(lambda: self.data.df['fvg_bullish_bottom'])
        self.fvg_bearish = self.I(lambda: self.data.df['fvg_bearish'])
        self.fvg_bearish_top = self.I(lambda: self.data.df['fvg_bearish_top'])
        self.fvg_bearish_bottom = self.I(lambda: self.data.df['fvg_bearish_bottom'])
        self.looking_for_pullback_long = False
        self.looking_for_pullback_short = False
        self.last_swing_high = None
        self.last_swing_low = None

        # Trade management variables
        self.breakout_start_price = None
        self.breakout_end_price = None
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False

    def next(self):
        volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]

        # --- Position Management (Take Profit) ---
        if self.position:
            if self.breakout_end_price is None or self.breakout_start_price is None:
                return # Not enough data to manage TP yet

            fib_range = abs(self.breakout_end_price - self.breakout_start_price)
            if self.position.is_long:
                tp1 = self.breakout_end_price + (fib_range * 0.5)
                tp2 = self.breakout_end_price + (fib_range * 0.618)

                if not self.tp1_hit and self.data.High[-1] >= tp1:
                    self.position.close(portion=0.33)
                    self.tp1_hit = True
                if not self.tp2_hit and self.data.High[-1] >= tp2:
                    self.position.close(portion=0.5) # Close 50% of remaining
                    self.tp2_hit = True

            elif self.position.is_short:
                tp1 = self.breakout_end_price - (fib_range * 0.5)
                tp2 = self.breakout_end_price - (fib_range * 0.618)

                if not self.tp1_hit and self.data.Low[-1] <= tp1:
                    self.position.close(portion=0.33)
                    self.tp1_hit = True
                if not self.tp2_hit and self.data.Low[-1] <= tp2:
                    self.position.close(portion=0.5) # Close 50% of remaining
                    self.tp2_hit = True
            return # Don't look for new trades while in a position

        # --- State Updates ---
        if self.swing_high[-1]: self.last_swing_high = self.data.High[-1]
        if self.swing_low[-1]: self.last_swing_low = self.data.Low[-1]

        # --- Entry Logic ---
        if self.looking_for_pullback_long:
            if self.fvg_bullish[-2] and self.data.Low[-1] <= self.fvg_bullish_top[-2] and volume_confirmed:
                sl = self.data.Close[-1] - (self.atr[-1] * self.atr_sl_multiplier)
                self.buy(sl=sl)
                self.looking_for_pullback_long = False

        elif self.htf_uptrend[-1] and self.last_swing_high is not None:
            if self.data.Close[-1] > self.last_swing_high:
                self.looking_for_pullback_long = True
                self.breakout_start_price = self.last_swing_low # Prior swing low
                self.breakout_end_price = self.data.High[-1]
                self.last_swing_high = None
                self.tp1_hit = self.tp2_hit = False # Reset TP flags

        if self.looking_for_pullback_short:
            if self.fvg_bearish[-2] and self.data.High[-1] >= self.fvg_bearish_bottom[-2] and volume_confirmed:
                sl = self.data.Close[-1] + (self.atr[-1] * self.atr_sl_multiplier)
                self.sell(sl=sl)
                self.looking_for_pullback_short = False

        elif not self.htf_uptrend[-1] and self.last_swing_low is not None:
            if self.data.Close[-1] < self.last_swing_low:
                self.looking_for_pullback_short = True
                self.breakout_start_price = self.last_swing_high # Prior swing high
                self.breakout_end_price = self.data.Low[-1]
                self.last_swing_low = None
                self.tp1_hit = self.tp2_hit = False # Reset TP flags

if __name__ == '__main__':
    # This harness is for development and will be adapted
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        data.columns = [col.strip().title() for col in data.columns]
    except FileNotFoundError:
        print("Data file not found. Falling back to synthetic data.")
        data = pd.DataFrame(index=pd.to_datetime(pd.date_range('2023-01-01', periods=3000, freq='15min')))
        data['Open'], data['High'], data['Low'], data['Close'], data['Volume'] = [np.random.rand(3000) for _ in range(5)]

    data_processed = preprocess_data(data.copy())
    bt = Backtest(data_processed, PullbackTradingStrategy, cash=100_000, commission=.002)
    stats = bt.run()
    print(stats)
