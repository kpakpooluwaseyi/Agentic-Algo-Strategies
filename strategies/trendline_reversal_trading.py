
import pandas as pd
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

def preprocess_data(df, distance, min_slope, max_slope):
    """
    Identifies swing points, extrapolates trendlines, and notes the last swing points
    for setting SL/TP and for trailing stops.
    """
    # 1. Find swing points
    high_peaks_idx, _ = find_peaks(df['High'], distance=distance)
    low_peaks_idx, _ = find_peaks(-df['Low'], distance=distance)

    df['swing_high'] = np.nan
    df.loc[df.index[high_peaks_idx], 'swing_high'] = df['High'].iloc[high_peaks_idx]
    df['swing_low'] = np.nan
    df.loc[df.index[low_peaks_idx], 'swing_low'] = df['Low'].iloc[low_peaks_idx]

    # Forward-fill the last known swing points
    df['last_swing_high'] = df['swing_high'].ffill()
    df['last_swing_low'] = df['swing_low'].ffill()

    # 2. Combine and sort all swing points
    peaks = []
    for idx in high_peaks_idx:
        peaks.append({'index': idx, 'type': 'high', 'price': df['High'].iloc[idx]})
    for idx in low_peaks_idx:
        peaks.append({'index': idx, 'type': 'low', 'price': df['Low'].iloc[idx]})
    peaks.sort(key=lambda x: x['index'])

    # 3. Filter for an alternating sequence
    alternating_peaks = []
    if peaks:
        alternating_peaks.append(peaks[0])
        for i in range(1, len(peaks)):
            if peaks[i]['type'] != alternating_peaks[-1]['type']:
                alternating_peaks.append(peaks[i])

    # 4. Find and extrapolate trendlines
    df['uptrend_line'] = np.nan
    df['downtrend_line'] = np.nan
    lows = [p for p in alternating_peaks if p['type'] == 'low']
    highs = [p for p in alternating_peaks if p['type'] == 'high']

    # UPTREND lines (higher lows)
    for i in range(len(lows) - 1):
        p1, p2 = lows[i], lows[i+1]
        if p2['price'] > p1['price'] and any(p1['index'] < h['index'] < p2['index'] for h in highs):
            slope = (p2['price'] - p1['price']) / (p2['index'] - p1['index'])
            if min_slope < slope < max_slope:
                for j in range(p2['index'], len(df)):
                    df.iat[j, df.columns.get_loc('uptrend_line')] = p2['price'] + slope * (j - p2['index'])

    # DOWNTREND lines (lower highs)
    for i in range(len(highs) - 1):
        p1, p2 = highs[i], highs[i+1]
        if p2['price'] < p1['price'] and any(p1['index'] < l['index'] < p2['index'] for l in lows):
            slope = (p2['price'] - p1['price']) / (p2['index'] - p1['index'])
            if -max_slope < slope < -min_slope:
                for j in range(p2['index'], len(df)):
                    df.iat[j, df.columns.get_loc('downtrend_line')] = p2['price'] + slope * (j - p2['index'])

    return df

class TrendlineReversalStrategy(Strategy):
    swing_distance = 15
    min_trendline_slope = 0.05
    max_trendline_slope = 0.8
    touch_tolerance_pct = 0.005
    rr_ratio = 1.5
    sl_buffer_pct = 0.01

    def init(self):
        self.uptrend_line = self.data.uptrend_line
        self.downtrend_line = self.data.downtrend_line
        self.last_swing_high = self.data.last_swing_high
        self.last_swing_low = self.data.last_swing_low

    def next(self):
        # --- TRAILING STOP LOGIC ---
        if self.position:
            if self.position.is_long:
                new_sl = self.last_swing_low[-1] * (1 - self.sl_buffer_pct)
                if new_sl > self.trades[0].sl and new_sl < self.data.Close[-1]:
                    self.trades[0].sl = new_sl
            else:
                new_sl = self.last_swing_high[-1] * (1 + self.sl_buffer_pct)
                if new_sl < self.trades[0].sl and new_sl > self.data.Close[-1]:
                    self.trades[0].sl = new_sl
            return

        # --- ENTRY LOGIC ---
        price = self.data.Close[-1]
        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]
        current_open = self.data.Open[-1]
        uptrend = self.uptrend_line[-1]
        downtrend = self.downtrend_line[-1]

        # LONG ENTRY
        if pd.notna(uptrend):
            if current_low <= uptrend * (1 + self.touch_tolerance_pct) and price > uptrend:
                if price > current_open:
                    sl = self.last_swing_low[-1] * (1 - self.sl_buffer_pct)
                    tp = self.last_swing_high[-1]
                    if sl < price and tp > price and (tp - price) / (price - sl) >= self.rr_ratio:
                        self.buy(sl=sl, tp=tp)
                        return

        # SHORT ENTRY
        if pd.notna(downtrend):
            if current_high >= downtrend * (1 - self.touch_tolerance_pct) and price < downtrend:
                if price < current_open:
                    sl = self.last_swing_high[-1] * (1 + self.sl_buffer_pct)
                    tp = self.last_swing_low[-1]
                    if sl > price and tp < price and (price - tp) / (sl - price) >= self.rr_ratio:
                        self.sell(sl=sl, tp=tp)
                        return

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        data.columns = [col.strip().capitalize() for col in data.columns]
        if 'Unnamed: 6' in data.columns:
            data.drop(columns=['Unnamed: 6'], inplace=True)

        print("Preprocessing data...")
        preprocessed_data = preprocess_data(data.copy(),
                                           distance=TrendlineReversalStrategy.swing_distance,
                                           min_slope=TrendlineReversalStrategy.min_trendline_slope,
                                           max_slope=TrendlineReversalStrategy.max_trendline_slope)
        preprocessed_data.dropna(subset=['last_swing_high', 'last_swing_low'], inplace=True)

        bt = Backtest(preprocessed_data, TrendlineReversalStrategy, cash=100_000, commission=.002)

        print("Running backtest...")
        stats = bt.run()
        print(stats)

        os.makedirs('results', exist_ok=True)
        plot_filename = 'results/trendline_reversal_trading_plot.html'
        json_filename = 'results/temp_result.json'

        try:
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Backtest plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")

        def sanitize_stats(stats_series):
            sanitized = {}
            for key, value in stats_series.items():
                if isinstance(value, (pd.Timestamp, pd.Timedelta)) or not isinstance(key, str): continue
                if pd.isna(value): sanitized[key] = None
                elif isinstance(value, (np.integer, np.int64)): sanitized[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)): sanitized[key] = float(value)
                elif isinstance(value, bool): sanitized[key] = bool(value)
                else: sanitized[key] = value
            sanitized['# Trades'] = sanitized.get('# Trades', 0)
            return sanitized

        # Convert stats to a dictionary and remove non-serializable objects
        stats_dict = stats.to_dict()
        if '_strategy' in stats_dict:
            del stats_dict['_strategy']
        if '_equity_curve' in stats_dict:
            del stats_dict['_equity_curve']
        if '_trades' in stats_dict:
            del stats_dict['_trades']

        final_stats = sanitize_stats(stats_dict)

        with open(json_filename, 'w') as f:
            json.dump(final_stats, f, indent=4)
        print(f"Backtest stats saved to {json_filename}")
