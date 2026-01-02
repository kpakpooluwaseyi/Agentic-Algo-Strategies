import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np
from scipy.signal import find_peaks
import json

def sanitize_json(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_json(i) for i in obj]
    if isinstance(obj, pd.DataFrame):
        return f"DataFrame with shape {obj.shape}"
    if isinstance(obj, pd.Series):
        return sanitize_json(obj.to_list())
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    try:
        return str(obj)
    except Exception:
        return "Unserializable object"

def calculate_volume_profile(df, bins=50):
    """
    Calculates daily volume profiles and identifies POC, HVNs, and LVNs.
    Maps these levels to the *next* day to avoid lookahead bias.
    """
    df['Date'] = df.index.date
    daily_levels = {}

    for date, group in df.groupby('Date'):
        price_range = (group['Low'].min(), group['High'].max())
        if price_range[1] == price_range[0]:
            continue

        hist, bin_edges = np.histogram(
            group['Close'], bins=bins, range=price_range, weights=group['Volume']
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks (HVNs) and the main peak (POC)
        peaks, _ = find_peaks(hist, distance=5)
        if len(peaks) > 0:
            poc_index = peaks[np.argmax(hist[peaks])]
            poc = bin_centers[poc_index]
            hvns = [bin_centers[p] for p in peaks if p != poc_index]
        else:
            poc = group['Close'].iloc[-1]
            hvns = []

        # Find valleys (LVNs) by inverting the histogram
        valleys, _ = find_peaks(-hist, distance=5)
        lvns = [bin_centers[v] for v in valleys]

        daily_levels[date] = {'poc': poc, 'hvns': hvns, 'lvns': lvns}

    levels_df = pd.DataFrame.from_dict(daily_levels, orient='index')
    # Shift levels to apply to the next day, avoiding lookahead bias
    levels_df = levels_df.shift(1)

    # Map the daily levels to the intraday DataFrame
    df['poc'] = df.index.normalize().map(levels_df['poc'])
    df['hvns'] = df.index.normalize().map(levels_df['hvns'])
    df['lvns'] = df.index.normalize().map(levels_df['lvns'])

    df.drop(columns=['Date'], inplace=True)

    # Forward fill the levels for each bar within the day
    df[['poc', 'hvns', 'lvns']] = df[['poc', 'hvns', 'lvns']].ffill()

    return df


class VolumeProfileLVNBounce(Strategy):
    rr_ratio = 1.5 # Adjusted R:R for more realistic targets
    sl_buffer_pct = 0.01

    def init(self):
        pass

    def next(self):
        if self.position:
            return

        current_price = self.data.Close[-1]

        # Ensure the level data is not NaN for the current bar
        if pd.isna(self.data.poc[-1]):
            return

        lvns = self.data.lvns[-1]
        hvns = self.data.hvns[-1]
        poc = self.data.poc[-1]

        if not lvns:
            return

        # --- Entry Logic ---

        # Long Entry: Bounce off an LVN from above
        for lvn in sorted(lvns, reverse=True):
            if self.data.Low[-1] <= lvn <= self.data.High[-1] and current_price > lvn:
                if self.data.Close[-1] > self.data.Open[-1]:
                    sl = lvn * (1 - self.sl_buffer_pct)

                    potential_tps = [p for p in hvns + [poc] if p > current_price]
                    if not potential_tps:
                        continue
                    tp = min(potential_tps)

                    if current_price > sl and tp > current_price and (tp - current_price) / (current_price - sl) >= self.rr_ratio:
                        self.buy(sl=sl, tp=tp)
                        return

        # Short Entry: Bounce off an LVN from below
        for lvn in sorted(lvns):
            if self.data.Low[-1] <= lvn <= self.data.High[-1] and current_price < lvn:
                if self.data.Close[-1] < self.data.Open[-1]:
                    sl = lvn * (1 + self.sl_buffer_pct)

                    potential_tps = [p for p in hvns + [poc] if p < current_price]
                    if not potential_tps:
                        continue
                    tp = max(potential_tps)

                    if current_price < sl and tp < current_price and (current_price - tp) / (sl - current_price) >= self.rr_ratio:
                        self.sell(sl=sl, tp=tp)
                        return

if __name__ == '__main__':
    data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    # Clean column names
    data.columns = [col.strip().capitalize() for col in data.columns]
    if 'Unnamed: 6' in data.columns:
        data.drop(columns=['Unnamed: 6'], inplace=True)

    data = calculate_volume_profile(data)
    data.dropna(inplace=True)

    bt = Backtest(data, VolumeProfileLVNBounce, cash=100000, commission=.002)
    stats = bt.run()

    print(stats)

    try:
        bt.plot(filename='results/volume_profile_lvn_bounce.html', open_browser=False)
    except Exception as e:
        print(f"Error plotting: {e}")

    sanitized_stats = sanitize_json(stats.to_dict())
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)
