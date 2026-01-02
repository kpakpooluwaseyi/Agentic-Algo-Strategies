import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from backtesting import Backtest, Strategy
import pandas_ta as ta
import json
import os

def _calculate_hvn_zones(df: pd.DataFrame, num_bins=50, prominence_percent=1.0) -> list:
    if df.empty: return []
    price_range = (df['Low'].min(), df['High'].max())
    bins = np.linspace(price_range[0], price_range[1], num_bins + 1)
    volume_per_bin = np.zeros(num_bins)
    binned_prices = np.digitize(df['Close'], bins) - 1
    for bin_idx in range(num_bins):
        volume_per_bin[bin_idx] = df['Volume'][binned_prices == bin_idx].sum()
    if volume_per_bin.max() == 0: return []
    prominence = volume_per_bin.max() * (prominence_percent / 100)
    peaks, _ = find_peaks(volume_per_bin, prominence=prominence)
    zone_prices = [bins[p] + (bins[1] - bins[0]) / 2 for p in peaks]
    return sorted(zone_prices)

def add_daily_poc(df: pd.DataFrame) -> pd.DataFrame:
    daily_data = df.groupby(df.index.date).agg(
        poc_price=('Close', lambda x: x.loc[df['Volume'].loc[x.index].idxmax()] if not x.empty else np.nan)
    )
    daily_data['prev_day_poc'] = daily_data['poc_price'].shift(1)
    df = df.join(daily_data[['prev_day_poc']], on=df.index.date)
    df['prev_day_poc'] = df['prev_day_poc'].bfill().ffill()
    return df

def is_bullish_engulfing(data):
    if len(data) < 2: return False
    return (data.Close[-2] < data.Open[-2] and
            data.Close[-1] > data.Open[-1] and
            data.Open[-1] <= data.Close[-2] and
            data.Close[-1] >= data.Open[-2])

def is_bearish_engulfing(data):
    if len(data) < 2: return False
    return (data.Close[-2] > data.Open[-2] and
            data.Close[-1] < data.Open[-1] and
            data.Open[-1] >= data.Close[-2] and
            data.Close[-1] <= data.Open[-2])

class InstitutionalVolumeZoneTrading(Strategy):
    atr_period = 14
    zone_width_multiplier = 0.5
    min_risk_reward_ratio = 1.5
    recalculation_period = 1000

    def init(self):
        self.atr = self.I(ta.atr, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), length=self.atr_period)
        self.prev_day_poc = self.I(lambda: self.data.prev_day_poc, name="prev_day_poc")
        self.hvn_zones = []
        self.last_recalculation = -self.recalculation_period

    def next(self):
        if len(self.data) - self.last_recalculation >= self.recalculation_period:
            if len(self.data) > 1:
                historical_data = self.data.df.iloc[:len(self.data)-1]
                self.hvn_zones = _calculate_hvn_zones(historical_data)
            self.last_recalculation = len(self.data)

        if self.position or not self.hvn_zones:
            return

        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]
        prev_poc = self.prev_day_poc[-1]

        if np.isnan(prev_poc) or np.isnan(current_atr):
            return

        zones_below = [z for z in self.hvn_zones if z < current_price]
        zones_above = [z for z in self.hvn_zones if z > current_price]
        support_hvn = max(zones_below) if zones_below else None
        resistance_hvn = min(zones_above) if zones_above else None

        if support_hvn and resistance_hvn and abs(support_hvn - prev_poc) <= current_atr:
            if abs(current_price - support_hvn) <= current_atr * self.zone_width_multiplier:
                if is_bullish_engulfing(self.data):
                    sl = support_hvn - current_atr
                    tp = resistance_hvn
                    if (current_price - sl) > 0 and (tp - current_price) / (current_price - sl) >= self.min_risk_reward_ratio:
                        self.buy(sl=sl, tp=tp)

        if resistance_hvn and support_hvn and abs(resistance_hvn - prev_poc) <= current_atr:
            if abs(current_price - resistance_hvn) <= current_atr * self.zone_width_multiplier:
                if is_bearish_engulfing(self.data):
                    sl = resistance_hvn + current_atr
                    tp = support_hvn
                    if (sl - current_price) > 0 and (current_price - tp) / (sl - current_price) >= self.min_risk_reward_ratio:
                        self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
        data.columns = [c.strip().title() for c in data.columns]

        data = add_daily_poc(data)
        data.dropna(subset=['prev_day_poc'], inplace=True)

        if data.empty:
            raise ValueError("Data is empty after preprocessing.")

        bt = Backtest(data, InstitutionalVolumeZoneTrading, cash=100000, commission=.002)

        print("Running final backtest...")
        stats = bt.run()
        print(stats)

        os.makedirs('results', exist_ok=True)

        def sanitize_stats(stats):
            sanitized = {}
            for key, value in stats.items():
                if isinstance(value, (np.integer, np.floating, int, float)):
                    sanitized[key] = float(value) if np.isfinite(value) else None
                elif isinstance(value, pd.Timestamp):
                    sanitized[key] = value.isoformat()
                elif isinstance(value, pd.Timedelta):
                     sanitized[key] = str(value)
                elif not isinstance(value, (pd.Series, pd.DataFrame)):
                    sanitized[key] = str(value)
            return sanitized

        result_to_save = sanitize_stats(stats)

        with open('results/temp_result.json', 'w') as f:
            json.dump(result_to_save, f, indent=4)

        print("\nBacktest results saved to results/temp_result.json")

        plot_filename = 'results/institutional_volume_zone_trading.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")

    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found.")
    except Exception as e:
        print(f"An error occurred during backtesting: {e}")
