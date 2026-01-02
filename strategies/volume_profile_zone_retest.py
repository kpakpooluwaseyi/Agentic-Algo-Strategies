
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import json
import os

def calculate_daily_volume_profile(daily_data, n_bins=100):
    """
    Calculates a simplified volume profile for a given day's data.

    Returns:
        - point_of_control (poc)
        - value_area_high (va_high)
        - value_area_low (va_low)
    """
    if daily_data.empty:
        return np.nan, np.nan, np.nan

    price_range = daily_data['High'].max() - daily_data['Low'].min()
    if price_range == 0:
        return daily_data['Close'].iloc[-1], daily_data['High'].iloc[-1], daily_data['Low'].iloc[-1]

    bins = np.linspace(daily_data['Low'].min(), daily_data['High'].max(), n_bins + 1)

    volume_per_bin = np.zeros(n_bins)

    for _, row in daily_data.iterrows():
        price_start_bin = np.searchsorted(bins, row['Low'], side='right') - 1
        price_end_bin = np.searchsorted(bins, row['High'], side='left')

        # Clamp bins to valid range
        price_start_bin = max(0, price_start_bin)
        price_end_bin = min(n_bins, price_end_bin)

        if price_start_bin >= price_end_bin:
            if price_start_bin < n_bins:
                 volume_per_bin[price_start_bin] += row['Volume']
            continue

        num_bins_spanned = price_end_bin - price_start_bin
        volume_per_bin[price_start_bin:price_end_bin] += row['Volume'] / num_bins_spanned

    # Point of Control (POC)
    poc_index = np.argmax(volume_per_bin)
    poc = (bins[poc_index] + bins[poc_index + 1]) / 2

    # Value Area (VA) ~70%
    total_volume = np.sum(volume_per_bin)
    if total_volume == 0:
        return poc, daily_data['High'].max(), daily_data['Low'].min()

    target_volume = total_volume * 0.7

    # Expand around POC
    current_volume = volume_per_bin[poc_index]
    va_start_index, va_end_index = poc_index, poc_index + 1

    while current_volume < target_volume:
        left_index = va_start_index - 1
        right_index = va_end_index

        if left_index < 0 and right_index >= n_bins:
            break

        left_volume = volume_per_bin[left_index] if left_index >= 0 else -1
        right_volume = volume_per_bin[right_index] if right_index < n_bins else -1

        if left_volume > right_volume:
            current_volume += left_volume
            va_start_index -= 1
        else:
            current_volume += right_volume
            va_end_index += 1

    va_low = bins[max(0, va_start_index)]
    va_high = bins[min(n_bins, va_end_index)]

    return poc, va_high, va_low

def preprocess_data(df):
    """
    Applies the daily volume profile calculation to the entire dataset.
    Shifts the results to avoid lookahead bias.
    """
    df['date'] = df.index.date

    # Calculate VP for each day
    daily_vp = df.groupby('date').apply(calculate_daily_volume_profile)
    daily_vp = pd.DataFrame(daily_vp.tolist(), index=daily_vp.index, columns=['poc', 'va_high', 'va_low'])

    # Shift data to prevent lookahead bias (use previous day's VP)
    daily_vp = daily_vp.shift(1)

    # Merge back into the main dataframe, preserving the original index
    df = pd.merge(df, daily_vp, left_on='date', right_index=True, how='left')
    df = df.drop(columns=['date'])

    # Forward fill to cover weekends/holidays in data
    df[['poc', 'va_high', 'va_low']] = df[['poc', 'va_high', 'va_low']].ffill()

    # Drop rows where VP could not be calculated (typically the first day)
    df = df.dropna(subset=['poc', 'va_high', 'va_low'])

    return df

# Define states
SEEKING = 0
SEPARATED_ABOVE = 1
SEPARATED_BELOW = 2

class VolumeProfileZoneRetest(Strategy):
    """
    Strategy based on retesting high-volume zones (Value Areas).
    """
    separation_factor = 0.1  # How far away price must move (multiple of VA range)
    rr_ratio = 1.5           # Risk-reward ratio for TP

    def init(self):
        # Pre-calculated indicators
        self.poc = self.I(lambda x: x, self.data.df['poc'].values)
        self.va_high = self.I(lambda x: x, self.data.df['va_high'].values)
        self.va_low = self.I(lambda x: x, self.data.df['va_low'].values)

        # State machine
        self.state = SEEKING
        self.zone_tested_today = False
        self.current_day = -1

    def next(self):
        # Reset state at the start of a new day
        if self.data.index[-1].day != self.current_day:
            self.current_day = self.data.index[-1].day
            self.state = SEEKING
            self.zone_tested_today = False

        if self.position:
            return

        poc = self.poc[-1]
        va_high = self.va_high[-1]
        va_low = self.va_low[-1]
        va_range = va_high - va_low

        if va_range <= 0:
            return

        separation_distance = va_range * self.separation_factor

        # State: SEEKING
        # State: SEEKING
        if self.state == SEEKING:
            if self.data.High[-1] > va_high + separation_distance:
                self.state = SEPARATED_ABOVE
            elif self.data.Low[-1] < va_low - separation_distance:
                self.state = SEPARATED_BELOW

        # State: SEPARATED_ABOVE (Look for long entry)
        elif self.state == SEPARATED_ABOVE:
            if self.data.Low[-1] < va_low: # Price has moved through the zone, reset
                self.state = SEEKING
            elif not self.zone_tested_today and self.data.Low[-1] <= va_high:
                entry_price = va_high
                sl = va_low # Stop-loss at the bottom of the value area

                if sl < entry_price:
                    tp = entry_price + (entry_price - sl) * self.rr_ratio
                    if tp > entry_price:
                        self.buy(limit=entry_price, sl=sl, tp=tp)
                        self.zone_tested_today = True

        # State: SEPARATED_BELOW (Look for short entry)
        elif self.state == SEPARATED_BELOW:
            if self.data.High[-1] > va_high: # Price has moved through the zone, reset
                self.state = SEEKING
            elif not self.zone_tested_today and self.data.High[-1] >= va_low:
                entry_price = va_low
                sl = va_high # Stop-loss at the top of the value area

                if sl > entry_price:
                    tp = entry_price - (sl - entry_price) * self.rr_ratio
                    if tp < entry_price:
                        self.sell(limit=entry_price, sl=sl, tp=tp)
                        self.zone_tested_today = True

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print("Data file not found. Please place it at data/BTC-USD-15m.csv")
    else:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.strip().title() for c in data.columns]

        # Preprocess the data to include volume profile info
        data = preprocess_data(data)

        bt = Backtest(data, VolumeProfileZoneRetest, cash=100_000, commission=.002)

        stats = bt.run()
        print(stats)

        # Save results
        os.makedirs('results', exist_ok=True)

        def sanitize_stats(stats):
            """Prepares stats object for JSON serialization."""
            clean_stats = {}
            for key, value in stats.items():
                if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    continue
                if isinstance(value, (np.integer, np.floating)):
                    clean_stats[key] = float(value)
                elif isinstance(value, (int, float, bool, str)) or value is None:
                    clean_stats[key] = value
            return clean_stats

        # Sanitize and save main stats
        result_to_save = sanitize_stats(stats)
        with open('results/temp_result.json', 'w') as f:
            json.dump(result_to_save, f, indent=4)

        print("Backtest results saved to results/temp_result.json")

        try:
            bt.plot(filename='results/volume_profile_zone_retest.html', open_browser=False)
            print("Plot saved to results/volume_profile_zone_retest.html")
        except Exception as e:
            print(f"Could not generate plot: {e}")
