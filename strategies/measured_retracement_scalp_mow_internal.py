from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os
from scipy.signal import find_peaks

def generate_synthetic_data(num_patterns=10, bars_per_pattern=240, noise_level=0.2):
    close_prices = []
    base_price = 100
    rng = np.random.default_rng(seed=42)
    for i in range(num_patterns):
        if i % 2 == 0:
            p1 = np.linspace(base_price, base_price + 5, num=bars_per_pattern // 4)
            p2 = np.linspace(p1[-1], base_price + 2, num=bars_per_pattern // 4)
            p3 = np.linspace(p2[-1], p1[-1] - 1, num=bars_per_pattern // 4)
            p4 = np.linspace(p3[-1], base_price - 1, num=bars_per_pattern - 3 * (bars_per_pattern // 4))
        else:
            p1 = np.linspace(base_price, base_price - 5, num=bars_per_pattern // 4)
            p2 = np.linspace(p1[-1], base_price - 2, num=bars_per_pattern // 4)
            p3 = np.linspace(p2[-1], p1[-1] + 1, num=bars_per_pattern // 4)
            p4 = np.linspace(p3[-1], base_price + 1, num=bars_per_pattern - 3 * (bars_per_pattern // 4))
        pattern = np.concatenate([p1, p2, p3, p4])
        noise = rng.normal(0, noise_level, len(pattern))
        pattern_with_noise = pattern + noise
        close_prices.extend(pattern_with_noise)
        base_price = close_prices[-1]
    total_bars = len(close_prices)
    dates = pd.date_range(start='2023-01-01', periods=total_bars, freq='min')
    df = pd.DataFrame(index=dates, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    df['Close'] = close_prices
    price_diff = df['Close'].diff().fillna(0)
    df['Open'] = df['Close'] - price_diff
    df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0, noise_level, size=total_bars)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0, noise_level, size=total_bars)
    df['Volume'] = rng.integers(100, 1000, size=total_bars)
    df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    return df

def preprocess_data(df_1m):
    df_15m = df_1m.resample('15min').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    high_peaks_idx, _ = find_peaks(df_15m['High'], distance=5, prominence=1)
    low_troughs_idx, _ = find_peaks(-df_15m['Low'], distance=5, prominence=1)
    df_15m['swing_high'] = np.nan
    df_15m['swing_low'] = np.nan
    df_15m.iloc[high_peaks_idx, df_15m.columns.get_loc('swing_high')] = df_15m['High'].iloc[high_peaks_idx]
    df_15m.iloc[low_troughs_idx, df_15m.columns.get_loc('swing_low')] = df_15m['Low'].iloc[low_troughs_idx]
    df_15m['swing_high'] = df_15m['swing_high'].ffill()
    df_15m['swing_low'] = df_15m['swing_low'].ffill()

    # Calculate all Fibonacci levels
    fib_levels = [0.382, 0.50, 0.618, 0.786]
    for level in fib_levels:
        df_15m[f'aoi_{int(level*1000)}'] = np.nan
    df_15m['aoi_high'] = np.nan
    df_15m['aoi_low'] = np.nan

    new_low_after_high = (df_15m['swing_low'] != df_15m['swing_low'].shift(1)) & (df_15m['swing_high'] == df_15m['swing_high'].shift(1))
    indices_to_update = df_15m.index[new_low_after_high]

    if not indices_to_update.empty:
        aoi_high = df_15m.loc[indices_to_update, 'swing_high']
        aoi_low = df_15m.loc[indices_to_update, 'swing_low']
        price_range = aoi_high - aoi_low

        df_15m.loc[indices_to_update, 'aoi_high'] = aoi_high
        df_15m.loc[indices_to_update, 'aoi_low'] = aoi_low
        for level in fib_levels:
            df_15m.loc[indices_to_update, f'aoi_{int(level*1000)}'] = aoi_low + price_range * level

    # Forward fill all new columns
    ffill_cols = ['aoi_high', 'aoi_low'] + [f'aoi_{int(level*1000)}' for level in fib_levels]
    df_15m[ffill_cols] = df_15m[ffill_cols].ffill()

    # Merge into 1M data
    df_merged = pd.merge(df_1m, df_15m[ffill_cols], left_index=True, right_index=True, how='left')
    df_merged[ffill_cols] = df_merged[ffill_cols].ffill()

    return df_merged.dropna()

class MeasuredRetracementScalpMowInternalStrategy(Strategy):
    risk_reward_ratio = 5.0

    def init(self):
        self.setup_active = False
        self.entry_peak = 0
        self.current_aoi_low = 0
        self.setup_invalidated = False

    def next(self):
        if self.position:
            return

        current_high = self.data.High[-1]
        aoi_low = self.data.aoi_low[-1]

        # Invalidate setup if a new 15M swing structure has formed
        if self.current_aoi_low != aoi_low:
            self.setup_active = False
            self.setup_invalidated = False
            self.current_aoi_low = aoi_low

        # If setup was invalidated in a previous bar, do nothing until new structure
        if self.setup_invalidated:
            return

        aoi_382 = self.data.aoi_382[-1]
        aoi_500 = self.data.aoi_500[-1]
        aoi_618 = self.data.aoi_618[-1]
        aoi_786 = self.data.aoi_786[-1]

        # Activate setup when price enters the AOI (crosses above 50% level)
        if not self.setup_active and self.data.Close[-1] > aoi_500:
            self.setup_active = True
            self.entry_peak = current_high
            return

        if self.setup_active:
            self.entry_peak = max(self.entry_peak, current_high)

            # Invalidation logic: Check for rejection of other Fib levels
            # A rejection is a high that touches a level but closes below it.
            if (aoi_382 < current_high < aoi_500) or \
               (aoi_618 < current_high < aoi_786 and self.data.Close[-1] < aoi_618) or \
               (current_high > aoi_786 and self.data.Close[-1] < aoi_786):
                self.setup_invalidated = True
                self.setup_active = False
                return

            # Entry condition: Bearish candle in the 50% AOI
            is_bearish_candle = self.data.Close[-1] < self.data.Open[-1]
            if is_bearish_candle and self.data.High[-1] > aoi_500:
                stop_loss = self.entry_peak * 1.001
                take_profit = self.entry_peak - (self.entry_peak - aoi_low) * 0.5

                if take_profit >= self.data.Close[-1]:
                    self.setup_active = False
                    return

                risk = stop_loss - self.data.Close[-1]
                reward = self.data.Close[-1] - take_profit

                if risk > 0 and (reward / risk) >= self.risk_reward_ratio:
                    self.sell(sl=stop_loss, tp=take_profit)

                self.setup_active = False

def clean_stat(value):
    if value is None or np.isnan(value) or np.isinf(value): return None
    if isinstance(value, (np.int64, np.int32)): return int(value)
    if isinstance(value, (np.float64, np.float32)): return float(value)
    return value

if __name__ == '__main__':
    data_1m = generate_synthetic_data(num_patterns=20)
    data_processed = preprocess_data(data_1m)

    bt = Backtest(data_processed, MeasuredRetracementScalpMowInternalStrategy,
                  cash=100_000, commission=.002, finalize_trades=True)

    stats = bt.optimize(risk_reward_ratio=np.arange(2.0, 10.5, 0.5).tolist(),
                        maximize='Sharpe Ratio',
                        constraint=lambda p: p.risk_reward_ratio > 1)

    print("--- Best Run Stats ---")
    print(stats)

    os.makedirs('results', exist_ok=True)

    with open('results/temp_result.json', 'w') as f:
        json_stats = {'strategy_name': 'measured_retracement_scalp_mow_internal',
                      'return': clean_stat(stats.get('Return [%]')),
                      'sharpe': clean_stat(stats.get('Sharpe Ratio')),
                      'max_drawdown': clean_stat(stats.get('Max. Drawdown [%]')),
                      'win_rate': clean_stat(stats.get('Win Rate [%]')),
                      'total_trades': clean_stat(stats.get('# Trades', 0))}
        json.dump(json_stats, f, indent=2)
    print("\nResults saved to results/temp_result.json")

    plot_filename = 'results/measured_retracement_scalp_mow_internal_plot.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")
