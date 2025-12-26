
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import json
import os

def find_swings(data, prominence=5):
    """
    Finds alternating swing highs and lows in a given data series.
    `data` should be a pandas Series or NumPy array.
    """
    high_peaks, _ = find_peaks(data, prominence=prominence)
    low_peaks, _ = find_peaks(-data, prominence=prominence)

    all_peaks = [(i, 'high') for i in high_peaks] + [(i, 'low') for i in low_peaks]
    all_peaks.sort(key=lambda x: x[0])

    alternating_swings = []
    if not all_peaks:
        return np.array([])

    alternating_swings.append(all_peaks[0][0])
    last_type = all_peaks[0][1]

    for i in range(1, len(all_peaks)):
        current_index, current_type = all_peaks[i]
        if current_type != last_type:
            alternating_swings.append(current_index)
            last_type = current_type

    return np.array(alternating_swings, dtype=int)


def find_impulse_waves(data, swings):
    """
    Identifies the MOST RECENT 5-point impulse wave pattern from swing points.
    """
    prices = np.asarray(data)

    # Iterate backwards to find the most recent valid wave
    for i in range(len(swings) - 6, -1, -1):
        p_indices = swings[i:i+6]
        p_prices = prices[p_indices]

        p0_idx, p1_idx, p2_idx, p3_idx, p4_idx, p5_idx = p_indices
        p0, p1, p2, p3, p4, p5 = p_prices

        # Basic structure check for an uptrend impulse wave
        if not (p0 < p1 and p2 < p1 and p2 < p3 and p4 < p3 and p4 < p5 and p1 < p3 and p1 < p5):
            continue

        # Rule 2: Wave 3 is the longest, never the shortest.
        if not ((p3 - p2) > (p1 - p0) and (p3 - p2) > (p5 - p4)):
            continue

        # Rule 3: Wave 2 does not retrace beyond the start of Wave 1.
        if not (p2 > p0):
            continue

        # Rule 4: Wave 4 does not overlap with Wave 1.
        if not (p4 > p1):
            continue

        # If all rules pass, this is our most recent valid wave
        return {
            'indices': [p0_idx, p1_idx, p2_idx, p3_idx, p4_idx, p5_idx],
            'prices': [p0, p1, p2, p3, p4, p5],
            'wave4_zone': (p4, p3) # (low, high) of the corrective zone
        }
    return None # No wave found

class ElliottWaveCorrectiveZoneEntry(Strategy):
    swing_prominence = 15
    volume_lookback = 10
    risk_reward_ratio = 2.0
    wave_lookback = 250 # Bars to look back for a pattern

    def init(self):
        self.active_zone = None
        self.last_wave_end_idx = -1

    def is_bullish_engulfing(self):
        if len(self.data.Close) < 2: return False
        prev_open, prev_close = self.data.Open[-2], self.data.Close[-2]
        curr_open, curr_close = self.data.Open[-1], self.data.Close[-1]
        return (prev_close < prev_open and
                curr_close > curr_open and
                curr_open <= prev_close and
                curr_close > prev_open)

    def is_volume_drying_up(self):
        if len(self.data.Volume) < self.volume_lookback: return False
        avg_vol = np.mean(self.data.Volume[-self.volume_lookback:-1])
        return self.data.Volume[-1] < avg_vol * 0.75

    def next(self):
        # Only search for new patterns if we are not in a trade or tracking a zone
        if not self.position and self.active_zone is None:
            # Look at the last `wave_lookback` bars of data
            lookback_data = self.data.Close[-self.wave_lookback:]

            # Find swings within this lookback window
            swings = find_swings(lookback_data, self.swing_prominence)

            if len(swings) >= 6:
                # Find the most recent impulse wave
                wave = find_impulse_waves(lookback_data, swings)

                # If a new wave is found, set it as the active zone to trade
                if wave and wave['indices'][-1] > self.last_wave_end_idx:
                    self.active_zone = wave
                    self.last_wave_end_idx = wave['indices'][-1]

        # If we are tracking an active zone for entry
        if self.active_zone:
            current_price = self.data.Close[-1]
            zone_low, zone_high = self.active_zone['wave4_zone']

            # Check for entry conditions inside the zone
            if zone_low <= current_price <= zone_high:
                if self.is_volume_drying_up() and self.is_bullish_engulfing():
                    sl = zone_low * 0.995 # SL just below the zone
                    tp = current_price + (current_price - sl) * self.risk_reward_ratio

                    if tp > current_price and sl < current_price:
                       self.buy(sl=sl, tp=tp)
                       self.active_zone = None # Consume the zone after trading

            # Invalidate zone if price drops below it before entry
            elif current_price < zone_low:
                self.active_zone = None


if __name__ == '__main__':
    data_path = 'data/crypto/BTC-USD-15m.csv'
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.strip().capitalize() for c in data.columns]
    data = data.rename(columns={'Volume,': 'Volume'})


    bt = Backtest(data, ElliottWaveCorrectiveZoneEntry, cash=100000, commission=.002)

    stats = bt.run()
    print(stats)

    # --- Save results ---
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                sanitized[key] = None
            elif pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                sanitized[key] = float(value)
            else:
                sanitized[key] = value
        return sanitized

    clean_stats = sanitize_stats(stats)

    result_data = {
        'strategy_name': 'elliott_wave_corrective_zone_entry',
        'return': clean_stats.get('Return [%]'),
        'sharpe': clean_stats.get('Sharpe Ratio'),
        'max_drawdown': clean_stats.get('Max. Drawdown [%]'),
        'win_rate': clean_stats.get('Win Rate [%]'),
        'total_trades': clean_stats.get('# Trades')
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_data, f, indent=2)
        f.write('\n') # Add newline for POSIX compliance

    print("Backtest results saved to results/temp_result.json")

    try:
        bt.plot(filename='results/elliott_wave_corrective_zone_entry.html')
    except Exception as e:
        print(f"Could not generate plot: {e}")
