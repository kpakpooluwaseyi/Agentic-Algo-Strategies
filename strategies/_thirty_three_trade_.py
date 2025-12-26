
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from scipy.signal import find_peaks
import json
import os

def find_alternating_peaks(series, distance):
    """
    Finds alternating high and low peaks in a pandas Series.
    """
    high_peaks_indices, _ = find_peaks(series, distance=distance)
    low_peaks_indices, _ = find_peaks(-series, distance=distance)

    high_peaks = pd.DataFrame({'index': series.index[high_peaks_indices], 'type': 'high'})
    low_peaks = pd.DataFrame({'index': series.index[low_peaks_indices], 'type': 'low'})

    all_peaks = pd.concat([high_peaks, low_peaks]).sort_values(by='index').reset_index(drop=True)
    # Remove consecutive peaks of the same type
    all_peaks = all_peaks.loc[all_peaks['type'].shift() != all_peaks['type']]
    return all_peaks

def preprocess_data(df, daily_peak_distance=5, intraday_peak_distance=10):
    """
    Identifies 'Day 3' exhaustion patterns and intraday swing points.
    """
    # === Multi-Day Level Analysis ===
    daily_df = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

    if len(daily_df) >= daily_peak_distance * 6:
        daily_peaks = find_alternating_peaks(daily_df['Close'], distance=daily_peak_distance)
        daily_df['is_day3_top'] = False
        daily_df['is_day3_bottom'] = False

        for i in range(len(daily_peaks) - 5):
            window = daily_peaks.iloc[i:i+6]
            if list(window['type']) == ['low', 'high', 'low', 'high', 'low', 'high']:
                lows = daily_df['Close'][window[window['type']=='low']['index']]
                highs = daily_df['Close'][window[window['type']=='high']['index']]
                if highs.is_monotonic_increasing and lows.is_monotonic_increasing:
                    peak_date = window.iloc[-1]['index']
                    daily_df.loc[peak_date, 'is_day3_top'] = True

            if list(window['type']) == ['high', 'low', 'high', 'low', 'high', 'low']:
                highs = daily_df['Close'][window[window['type']=='high']['index']]
                lows = daily_df['Close'][window[window['type']=='low']['index']]
                if highs.is_monotonic_decreasing and lows.is_monotonic_decreasing:
                    peak_date = window.iloc[-1]['index']
                    daily_df.loc[peak_date, 'is_day3_bottom'] = True

        df['is_day3_top_day'] = df.index.normalize().map(daily_df['is_day3_top']).fillna(False)
        df['is_day3_bottom_day'] = df.index.normalize().map(daily_df['is_day3_bottom']).fillna(False)
    else:
        df['is_day3_top_day'] = False
        df['is_day3_bottom_day'] = False

    # === Intraday Level Analysis ===
    intraday_peaks = find_alternating_peaks(df['Close'], distance=intraday_peak_distance)
    df['is_intraday_high'] = False
    df['is_intraday_low'] = False
    df.loc[intraday_peaks[intraday_peaks['type']=='high']['index'], 'is_intraday_high'] = True
    df.loc[intraday_peaks[intraday_peaks['type']=='low']['index'], 'is_intraday_low'] = True

    # Add day identifier for state resets
    df['day_id'] = df.index.dayofyear

    return df

def passthrough(series):
    return series

class ThirtyThreeTradeStrategy(Strategy):
    intraday_peak_distance = 10
    risk_reward_ratio = 2.0
    sl_buffer_pct = 0.01

    def init(self):
        self.is_day3_top_day = self.I(passthrough, self.data.df['is_day3_top_day'].values, name="is_day3_top_day")
        self.is_day3_bottom_day = self.I(passthrough, self.data.df['is_day3_bottom_day'].values, name="is_day3_bottom_day")
        self.is_intraday_high = self.I(passthrough, self.data.df['is_intraday_high'].values, name="is_intraday_high")
        self.is_intraday_low = self.I(passthrough, self.data.df['is_intraday_low'].values, name="is_intraday_low")
        self.day_id = self.I(passthrough, self.data.df['day_id'].values, name="day_id")

        self.current_day_id = -1
        self.intraday_level_count = 0
        self.last_peak_type = None
        self.setup_type = None

    def is_bearish_engulfing(self):
        if len(self.data.Close) < 2: return False
        return (self.data.Close[-2] > self.data.Open[-2] and # Previous is bullish
                self.data.Close[-1] < self.data.Open[-1] and # Current is bearish
                self.data.Open[-1] >= self.data.Close[-2] and
                self.data.Close[-1] < self.data.Open[-2])

    def is_bullish_engulfing(self):
        if len(self.data.Close) < 2: return False
        return (self.data.Close[-2] < self.data.Open[-2] and # Previous is bearish
                self.data.Close[-1] > self.data.Open[-1] and # Current is bullish
                self.data.Open[-1] <= self.data.Close[-2] and
                self.data.Close[-1] > self.data.Open[-2])

    def next(self):
        # Daily state reset
        if self.day_id[-1] != self.current_day_id:
            self.current_day_id = self.day_id[-1]
            self.intraday_level_count = 0
            self.last_peak_type = None
            self.setup_type = None
            if self.is_day3_top_day[-1]:
                self.setup_type = 'sell'
            elif self.is_day3_bottom_day[-1]:
                self.setup_type = 'buy'

        if self.position or not self.setup_type:
            return

        # Count intraday levels
        if self.is_intraday_high[-1] and self.last_peak_type != 'high':
            self.intraday_level_count += 1
            self.last_peak_type = 'high'
        elif self.is_intraday_low[-1] and self.last_peak_type != 'low':
            self.intraday_level_count += 1
            self.last_peak_type = 'low'

        # Entry logic
        if self.intraday_level_count >= 6: # 3 highs and 3 lows
            if self.setup_type == 'sell' and self.is_bearish_engulfing():
                entry_price = self.data.Close[-1]
                sl = self.data.High[-1] * (1 + self.sl_buffer_pct)
                tp = entry_price - (sl - entry_price) * self.risk_reward_ratio
                if tp > 0 and sl > entry_price:
                    self.sell(sl=sl, tp=tp)
                    self.setup_type = None # Prevent re-entry

            elif self.setup_type == 'buy' and self.is_bullish_engulfing():
                entry_price = self.data.Close[-1]
                sl = self.data.Low[-1] * (1 - self.sl_buffer_pct)
                tp = entry_price + (entry_price - sl) * self.risk_reward_ratio
                if tp > entry_price and sl < entry_price:
                    self.buy(sl=sl, tp=tp)
                    self.setup_type = None # Prevent re-entry

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Clean column names
    data.columns = [col.strip().title() for col in data.columns]
    data = preprocess_data(data)

    bt = Backtest(data, ThirtyThreeTradeStrategy, cash=100_000, commission=.002)

    stats = bt.run()

    print(stats)

    os.makedirs('results', exist_ok=True)

    # Save stats to JSON
    stats_dict = dict(stats)
    # Remove non-serializable items
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)

    bt.plot(filename='results/_thirty_three_trade_.html')
