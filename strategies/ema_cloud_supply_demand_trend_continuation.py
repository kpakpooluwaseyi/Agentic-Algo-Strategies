import json
import pandas as pd
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks
import os

def EMA(series, period):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    return ta.ema(series, length=period)

def sanitize_stats(stats):
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = value.item()
        elif isinstance(value, (pd.Series, pd.DataFrame, Strategy)):
            continue
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized

def preprocess_data(df, htf='1h'):
    df_htf = df.resample(htf).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_htf['HTF_EMA_fast'] = EMA(df_htf['Close'], 20)
    df_htf['HTF_EMA_slow'] = EMA(df_htf['Close'], 50)
    df = df.join(df_htf[['HTF_EMA_fast', 'HTF_EMA_slow']], how='left')
    df[['HTF_EMA_fast', 'HTF_EMA_slow']] = df[['HTF_EMA_fast', 'HTF_EMA_slow']].ffill()
    return df

class EmaCloudSupplyDemand(Strategy):
    ema_fast_period = 20
    ema_slow_period = 50
    swing_lookback = 100
    peak_distance = 10
    min_rr = 1.5
    sl_buffer_pct = 0.01

    def init(self):
        self.ema_fast = self.I(EMA, self.data.Close, self.ema_fast_period)
        self.ema_slow = self.I(EMA, self.data.Close, self.ema_slow_period)
        self.htf_ema_fast = self.I(lambda x: x, self.data.HTF_EMA_fast, plot=False)
        self.htf_ema_slow = self.I(lambda x: x, self.data.HTF_EMA_slow, plot=False)
        self.last_demand_zone = None
        self.last_supply_zone = None

    def next(self):
        current_index = len(self.data.Close) - 1
        if current_index < self.swing_lookback or pd.isna(self.htf_ema_fast[-1]):
            return

        lookback_highs = self.data.High[-self.swing_lookback:]
        peak_indices, _ = find_peaks(lookback_highs, distance=self.peak_distance)
        if peak_indices.size > 0:
            last_peak_idx = current_index - self.swing_lookback + 1 + peak_indices[-1]
            peak_candle_open = self.data.Open[last_peak_idx]
            peak_candle_close = self.data.Close[last_peak_idx]
            zone = (min(peak_candle_open, peak_candle_close), max(peak_candle_open, peak_candle_close))
            sl = self.data.High[last_peak_idx] * (1 + self.sl_buffer_pct)
            self.last_supply_zone = {'zone': zone, 'sl': sl}

        lookback_lows = self.data.Low[-self.swing_lookback:]
        trough_indices, _ = find_peaks(-lookback_lows, distance=self.peak_distance)
        if trough_indices.size > 0:
            last_trough_idx = current_index - self.swing_lookback + 1 + trough_indices[-1]
            trough_candle_open = self.data.Open[last_trough_idx]
            trough_candle_close = self.data.Close[last_trough_idx]
            zone = (min(trough_candle_open, trough_candle_close), max(trough_candle_open, trough_candle_close))
            sl = self.data.Low[last_trough_idx] * (1 - self.sl_buffer_pct)
            self.last_demand_zone = {'zone': zone, 'sl': sl}

        if self.position:
            return

        ltf_uptrend = self.ema_fast[-1] > self.ema_slow[-1]
        htf_uptrend = self.htf_ema_fast[-1] > self.htf_ema_slow[-1]
        is_uptrend = ltf_uptrend and htf_uptrend

        ltf_downtrend = self.ema_fast[-1] < self.ema_slow[-1]
        htf_downtrend = self.htf_ema_fast[-1] < self.htf_ema_slow[-1]
        is_downtrend = ltf_downtrend and htf_downtrend

        entry_price = self.data.Close[-1]
        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]

        if is_uptrend and self.last_demand_zone and self.last_supply_zone:
            demand_low, demand_high = self.last_demand_zone['zone']
            if current_low <= demand_high and self.data.Close[-1] > self.data.Open[-1]:
                sl = self.last_demand_zone['sl']
                tp = self.last_supply_zone['zone'][0]
                if tp > entry_price and (entry_price - sl) > 0:
                    if (tp - entry_price) / (entry_price - sl) >= self.min_rr:
                        self.buy(sl=sl, tp=tp)

        elif is_downtrend and self.last_supply_zone and self.last_demand_zone:
            supply_low, supply_high = self.last_supply_zone['zone']
            if current_high >= supply_low and self.data.Close[-1] < self.data.Open[-1]:
                sl = self.last_supply_zone['sl']
                tp = self.last_demand_zone['zone'][1]
                if tp < entry_price and (sl - entry_price) > 0:
                    if (entry_price - tp) / (sl - entry_price) >= self.min_rr:
                        self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    try:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        data.columns = [col.strip().capitalize() for col in data.columns]
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        from backtesting.test import EURUSD
        data = EURUSD.copy().iloc[-3000:]
        print("Using synthetic EURUSD data as a fallback.")

    data = preprocess_data(data)

    bt = Backtest(data, EmaCloudSupplyDemand, cash=100000, commission=.002)
    stats = bt.run()
    print(stats)

    results_path = os.path.join(results_dir, 'temp_result.json')
    sanitized = sanitize_stats(stats)
    with open(results_path, 'w') as f:
        json.dump(sanitized, f, indent=4)
    print(f"Results saved to {results_path}")

    plot_path = os.path.join(results_dir, 'ema_cloud_supply_demand_trend_continuation.html')
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
