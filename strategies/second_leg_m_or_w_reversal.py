import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.signal import find_peaks
from backtesting import Strategy, Backtest
import json
from pandas import Timestamp, NaT, DataFrame, Series, Timedelta

def find_swing_points_scipy(high_prices, low_prices, distance):
    highs_np = np.asarray(high_prices)
    lows_np = np.asarray(low_prices)
    peaks, _ = find_peaks(highs_np, distance=distance)
    troughs, _ = find_peaks(-lows_np, distance=distance)
    swing_highs = np.full(len(highs_np), np.nan)
    swing_lows = np.full(len(lows_np), np.nan)
    swing_highs[peaks] = highs_np[peaks]
    swing_lows[troughs] = lows_np[troughs]
    return swing_highs, swing_lows

def sanitize_for_json(obj):
    if isinstance(obj, (DataFrame, Series)): return None
    if isinstance(obj, dict): return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [sanitize_for_json(i) for i in obj]
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, (Timestamp, pd.Timestamp)): return obj.isoformat()
    if isinstance(obj, Timedelta): return str(obj)
    if obj is NaT or pd.isna(obj): return None
    return obj

class SecondLegMWReversal(Strategy):
    # Strategy Parameters
    swing_lookback = 10
    min_leg_time_minutes = 30
    max_leg_time_minutes = 120
    stop_loss_pips = 100
    risk_reward_ratio = 2.0
    stop_hunt_distance = 50
    time_based_exit_bars = 8
    max_asia_range = 100
    london_start_hour = 8
    london_end_hour = 11
    risk_percent = 0.02 # Risk 2% of equity per trade

    def init(self):
        # ... (indicator setup remains the same)
        self.asia_high = self.data.asia_high
        self.asia_low = self.data.asia_low
        self.asia_range = self.data.asia_range
        self.swing_highs, self.swing_lows = self.I(
            find_swing_points_scipy, self.data.High, self.data.Low, self.swing_lookback
        )
        self.ema5 = self.I(ta.ema, pd.Series(self.data.Close), length=5)
        self.ema13 = self.I(ta.ema, pd.Series(self.data.Close), length=13)
        self.ema50 = self.I(ta.ema, pd.Series(self.data.Close), length=50)
        self.ema200 = self.I(ta.ema, pd.Series(self.data.Close), length=200)
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=14)
        self.m_state = 'SEARCHING'
        self.m_p1_price, self.m_p1_idx, self.m_p1_rsi = None, None, None
        self.w_state = 'SEARCHING'
        self.w_p1_price, self.w_p1_idx, self.w_p1_rsi = None, None, None


    def next(self):
        # ... (session logic remains the same)
        current_idx = len(self.data) - 1
        current_hour = self.data.index[-1].hour

        if self.position:
            if (current_idx - self.trades[0].entry_bar) >= self.time_based_exit_bars:
                self.position.close(comment="Time-based exit")
            return

        is_london_session = self.london_start_hour <= current_hour < self.london_end_hour
        is_asia_range_valid = self.asia_range[-1] < self.max_asia_range

        if is_london_session and is_asia_range_valid:
            self.manage_m_pattern(current_idx)
            self.manage_w_pattern(current_idx)


    def is_bearish_engulfing(self):
        return (self.data.Close[-1] < self.data.Open[-2] and
                self.data.Open[-1] > self.data.Close[-2] and
                self.data.Close[-2] > self.data.Open[-2])

    def is_bullish_engulfing(self):
        return (self.data.Close[-1] > self.data.Open[-2] and
                self.data.Open[-1] < self.data.Close[-2] and
                self.data.Close[-2] < self.data.Open[-2])

    def manage_m_pattern(self, current_idx):
        if self.m_state == 'SEARCHING':
            if (self.data.High[-1] > self.asia_high[-1] + self.stop_hunt_distance and
                    not np.isnan(self.swing_highs[-1])):
                self.m_state = 'LEG_1'
                self.m_p1_price = self.swing_highs[-1]
                self.m_p1_idx = current_idx
                self.m_p1_rsi = self.rsi[-1]
        elif self.m_state == 'LEG_1':
            if not np.isnan(self.swing_lows[-1]): self.m_state = 'CENTER'
            elif self.data.High[-1] > self.m_p1_price: self.m_state = 'SEARCHING'
        elif self.m_state == 'CENTER':
            if not np.isnan(self.swing_highs[-1]):
                p2_price, p2_idx = self.swing_highs[-1], current_idx
                p2_rsi = self.rsi[-1]
                time_diff = (p2_idx - self.m_p1_idx) * 15
                if (p2_price < self.m_p1_price and self.min_leg_time_minutes <= time_diff <= self.max_leg_time_minutes and
                    p2_rsi < self.m_p1_rsi and
                    self.ema5[-1] < self.ema13[-1] and
                    self.data.Close[-1] < self.ema50[-1] and self.data.Close[-1] < self.ema200[-1] and
                    self.is_bearish_engulfing()):
                    sl = max(self.m_p1_price, p2_price) + self.stop_loss_pips
                    tp = self.data.Close[-1] - (sl - self.data.Close[-1]) * self.risk_reward_ratio
                    size = self.equity * self.risk_percent / (sl - self.data.Close[-1])
                    if tp < self.data.Close[-1]: self.sell(sl=sl, tp=tp, size=size)
                self.m_state = 'SEARCHING'
            elif self.data.Close[-1] < self.swing_lows[-2] if len(self.swing_lows)>1 else self.data.Low[-2]:
                 self.m_state = 'SEARCHING'

    def manage_w_pattern(self, current_idx):
        if self.w_state == 'SEARCHING':
            if (self.data.Low[-1] < self.asia_low[-1] - self.stop_hunt_distance and
                    not np.isnan(self.swing_lows[-1])):
                self.w_state = 'LEG_1'
                self.w_p1_price = self.swing_lows[-1]
                self.w_p1_idx = current_idx
                self.w_p1_rsi = self.rsi[-1]
        elif self.w_state == 'LEG_1':
            if not np.isnan(self.swing_highs[-1]): self.w_state = 'CENTER'
            elif self.data.Low[-1] < self.w_p1_price: self.w_state = 'SEARCHING'
        elif self.w_state == 'CENTER':
            if not np.isnan(self.swing_lows[-1]):
                p2_price, p2_idx = self.swing_lows[-1], current_idx
                p2_rsi = self.rsi[-1]
                time_diff = (p2_idx - self.w_p1_idx) * 15
                if (p2_price > self.w_p1_price and self.min_leg_time_minutes <= time_diff <= self.max_leg_time_minutes and
                    p2_rsi > self.w_p1_rsi and
                    self.ema5[-1] > self.ema13[-1] and
                    self.data.Close[-1] > self.ema50[-1] and self.data.Close[-1] > self.ema200[-1] and
                    self.is_bullish_engulfing()):
                    sl = min(self.w_p1_price, p2_price) - self.stop_loss_pips
                    tp = self.data.Close[-1] + (self.data.Close[-1] - sl) * self.risk_reward_ratio
                    size = self.equity * self.risk_percent / (self.data.Close[-1] - sl)
                    if tp > self.data.Close[-1]: self.buy(sl=sl, tp=tp, size=size)
                self.w_state = 'SEARCHING'
            elif self.data.Close[-1] > self.swing_highs[-2] if len(self.swing_highs)>1 else self.data.High[-2]:
                 self.w_state = 'SEARCHING'

if __name__ == '__main__':
    # ... (data loading remains the same)
    data = pd.read_csv('data/crypto/BTC-USD-15m.csv', skipinitialspace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    if 'Unnamed: 6' in data.columns: data.drop(columns=['Unnamed: 6'], inplace=True)
    is_asia_session = (data.index.hour >= 0) & (data.index.hour < 8)
    daily_asia_range = data[is_asia_session].groupby(data[is_asia_session].index.date).agg(asia_high=('high', 'max'), asia_low=('low', 'min'))
    data['date'] = data.index.date
    data = pd.merge(data, daily_asia_range, left_on='date', right_index=True, how='left')
    data['asia_high'] = data['asia_high'].bfill().ffill()
    data['asia_low'] = data['asia_low'].bfill().ffill()
    data.drop(columns=['date'], inplace=True)
    data.dropna(inplace=True)
    data['asia_range'] = data['asia_high'] - data['asia_low']
    data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

    print("Data pre-processing complete.")
    bt = Backtest(data, SecondLegMWReversal, cash=100000, commission=.002)
    stats = bt.run()
    print(stats)
    bt.plot(filename="results/temp_plot.html")
    results_dict = sanitize_for_json(dict(stats))
    results_dict.pop('_strategy', None)
    results_dict.pop('_trades', None)
    results_dict.pop('_equity_curve', None)
    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    print("\nBacktest complete. Results saved.")
