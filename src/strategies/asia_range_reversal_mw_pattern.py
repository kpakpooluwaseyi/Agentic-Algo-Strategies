
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply
import pandas as pd
import numpy as np
import json
import os

# --- Helper Functions & Indicators ---

def is_hammer(df):
    op, hi, lo, cl = df.Open, df.High, df.Low, df.Close
    body = abs(cl - op)
    candle_range = hi - lo
    lower_wick = np.minimum(op, cl) - lo
    upper_wick = hi - np.maximum(op, cl)
    return (candle_range > 0) & (lower_wick > 2 * body) & (upper_wick < body)

def is_inverted_hammer(df):
    op, hi, lo, cl = df.Open, df.High, df.Low, df.Close
    body = abs(cl - op)
    candle_range = hi - lo
    lower_wick = np.minimum(op, cl) - lo
    upper_wick = hi - np.maximum(op, cl)
    return (candle_range > 0) & (upper_wick > 2 * body) & (lower_wick < body)

def is_tweezer_tops(df):
    hi, op, cl = df.High, df.Open, df.Close
    high_prev = hi.shift(1)
    is_similar_high = abs(hi - high_prev) / hi < 0.001
    is_first_bullish = cl.shift(1) > op.shift(1)
    is_second_bearish = cl < op
    return is_similar_high & is_first_bullish & is_second_bearish

def get_daily_touch_counts(data_df, hod, lod):
    df = pd.DataFrame({'High': data_df.High, 'Low': data_df.Low, 'hod': hod, 'lod': lod}, index=data_df.index)
    ny_time = df.index.tz_localize('UTC', nonexistent='shift_forward').tz_convert('America/New_York')
    df['date'] = ny_time.date

    df['touch_hod_event'] = (df.High >= df.hod) & ((df.High.shift(1) < df.hod.shift(1)) | df.hod.diff() != 0)
    df['touch_lod_event'] = (df.Low <= df.lod) & ((df.Low.shift(1) > df.lod.shift(1)) | df.lod.diff() != 0)

    hod_touches_today = df.groupby('date')['touch_hod_event'].cumsum()
    lod_touches_today = df.groupby('date')['touch_lod_event'].cumsum()
    return hod_touches_today.values, lod_touches_today.values

def mm_phase_indicator(close, ema_50, ema_200):
    phase = np.zeros_like(close)
    phase[close > ema_50] = 1
    phase[close < ema_50] = -1
    phase[(close > ema_200) & (close < ema_50)] = 2
    phase[(close < ema_200) & (close > ema_50)] = 2
    return phase

def ema_indicator(series, span):
    return series.ewm(span=span, adjust=False).mean()

class AsiaRangeReversalMwPatternStrategy(Strategy):
    pattern_lookback = 60
    proximity_percent = 0.5
    stop_hunt_threshold = 1.5

    def init(self):
        self.ny_time = self.data.index.tz_localize('UTC', nonexistent='shift_forward').tz_convert('America/New_York')

        self.ema50_1h = self.I(resample_apply, '1h', ema_indicator, self.data.Close, span=50)
        self.ema200_1h = self.I(resample_apply, '1h', ema_indicator, self.data.Close, span=200)
        self.mm_phase = self.I(mm_phase_indicator, self.data.Close, self.ema50_1h, self.ema200_1h)

        self.asia_hod, self.asia_lod = self._calculate_asia_levels()

        self.hammer = self.I(is_hammer, self.data.df)
        self.inverted_hammer = self.I(is_inverted_hammer, self.data.df)
        self.tweezer_tops = self.I(is_tweezer_tops, self.data.df)

        self.hod_touches, self.lod_touches = self.I(get_daily_touch_counts, self.data.df, self.asia_hod, self.asia_lod)
        self.atr = self.I(lambda: self.data.High - self.data.Low, name='ATR')

    def _calculate_asia_levels(self):
        df = self.data.df.copy()
        ny_time_series = pd.Series(self.ny_time, index=df.index)
        normalized_day = ny_time_series.dt.normalize()
        trading_day = np.where(ny_time_series.dt.hour < 20, normalized_day - pd.Timedelta(days=1), normalized_day)
        df['trading_day'] = pd.to_datetime(trading_day)
        asia_mask = (ny_time_series.dt.time >= pd.to_datetime('20:00').time()) | (ny_time_series.dt.time < pd.to_datetime('05:00').time())
        asia_data = df[asia_mask].copy()
        asia_data['body_min'] = asia_data[['Open', 'Close']].min(axis=1)
        asia_data['body_max'] = asia_data[['Open', 'Close']].max(axis=1)
        daily_levels = asia_data.groupby('trading_day').agg(asia_lod=('body_min', 'min'), asia_hod=('body_max', 'max'))
        merged = pd.merge(df, daily_levels, on='trading_day', how='left').set_index(df.index)
        return merged['asia_hod'].ffill().values, merged['asia_lod'].ffill().values

    def next(self):
        if len(self.data) < self.pattern_lookback: return
        current_time = self.ny_time[-1]

        if 17 <= current_time.hour < 20: return
        is_trading_window = (current_time.hour >= 20 and current_time.minute >= 30) or (current_time.hour < 10)
        if not is_trading_window: return

        if self.position:
            is_friday_afternoon = current_time.weekday() == 4 and current_time.hour >= 12
            if is_friday_afternoon and self._is_stop_hunt():
                self.position.close(comment="Friday Stop Hunt Protection")

        if not self.position:
            self._handle_long_entry()
            self._handle_short_entry()

    def _is_stop_hunt(self):
        if not self.position or len(self.atr) < 2: return False

        current_atr = self.atr[-1]
        avg_atr = np.mean(self.atr[-20:-1])
        is_large_spike = current_atr > avg_atr * self.stop_hunt_threshold

        if self.position.is_long:
            return is_large_spike and self.data.Close[-1] < self.data.Open[-1]
        else:
            return is_large_spike and self.data.Close[-1] > self.data.Open[-1]

    def _handle_long_entry(self):
        is_near_lod = self.data.Low[-1] <= self.asia_lod[-1] * (1 + self.proximity_percent / 100)
        has_touches = self.lod_touches[-1] >= 3
        is_pattern = self.hammer[-1] or self.inverted_hammer[-1]
        is_mm_phase = self.mm_phase[-1] in [2, -1]

        w_pattern_result, w_low = self._is_w_pattern()
        if is_near_lod and has_touches and is_pattern and is_mm_phase and w_pattern_result:
            sl = w_low * 0.998
            self.buy(sl=sl, size=0.5, tp=self.ema50_1h[-1])
            self.buy(sl=sl, size=0.5, tp=self.ema200_1h[-1])

    def _handle_short_entry(self):
        is_near_hod = self.data.High[-1] >= self.asia_hod[-1] * (1 - self.proximity_percent / 100)
        has_touches = self.hod_touches[-1] >= 3
        is_pattern = self.tweezer_tops[-1] or self.hammer[-1]
        is_mm_phase = self.mm_phase[-1] in [2, 1]

        m_pattern_result, m_high = self._is_m_pattern()
        if is_near_hod and has_touches and is_pattern and is_mm_phase and m_pattern_result:
            sl = m_high * 1.002
            self.sell(sl=sl, size=0.5, tp=self.ema50_1h[-1])
            self.sell(sl=sl, size=0.5, tp=self.ema200_1h[-1])

    def _is_w_pattern(self):
        lows = self.data.Low[-self.pattern_lookback:]
        if len(lows) < 5: return False, None
        troughs_idx = np.argpartition(lows, 2)[:2]
        if len(troughs_idx) < 2: return False, None

        first_trough_idx, second_trough_idx = np.sort(troughs_idx)
        first_trough_val, second_trough_val = lows[first_trough_idx], lows[second_trough_idx]

        is_valid = abs(first_trough_val - second_trough_val) / first_trough_val < 0.03
        formation_low = min(first_trough_val, second_trough_val)
        return is_valid, formation_low

    def _is_m_pattern(self):
        highs = self.data.High[-self.pattern_lookback:]
        if len(highs) < 5: return False, None
        peaks_idx = np.argpartition(-highs, 2)[:2]
        if len(peaks_idx) < 2: return False, None

        first_peak_idx, second_peak_idx = np.sort(peaks_idx)
        first_peak_val, second_peak_val = highs[first_peak_idx], highs[second_peak_idx]

        is_valid = abs(first_peak_val - second_peak_val) / first_peak_val < 0.03
        formation_high = max(first_peak_val, second_peak_val)
        return is_valid, formation_high

if __name__ == '__main__':
    def generate_synthetic_data(days=90, freq='5min'):
        n = int(days * 24 * (60 / int(freq.replace('min', ''))))
        dt = pd.to_datetime(pd.date_range('2023-01-01', periods=n, freq=freq))
        price_base = np.log(100 + np.sin(np.arange(n) / (24 * 30)) * 20)
        price_trend = price_base + np.arange(n) * 0.00001
        price = np.exp(price_trend + np.random.randn(n) * 0.001)

        data = pd.DataFrame({
            'Open': price, 'High': price * (1 + np.random.uniform(0, 0.001, n)),
            'Low': price * (1 - np.random.uniform(0, 0.001, n)),
            'Close': price + np.random.uniform(-0.0005, 0.0005, n) * price,
            'Volume': np.random.randint(100, 1000, n)}, index=dt)
        return data

    data = generate_synthetic_data()
    bt = Backtest(data, AsiaRangeReversalMwPatternStrategy, cash=100000, commission=.002)

    stats = bt.optimize(pattern_lookback=range(40, 80, 10), proximity_percent=[0.1, 0.3, 0.5], maximize='Sharpe Ratio')

    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'asia_range_reversal_mw_pattern',
            'return': float(stats['Return [%]']), 'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']), 'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['_trades'].shape[0]) if '# Trades' not in stats else int(stats['# Trades'])
        }, f, indent=2)

    bt.plot()
