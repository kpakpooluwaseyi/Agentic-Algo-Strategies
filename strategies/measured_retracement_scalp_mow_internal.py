import json
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from backtesting import Backtest, Strategy

# --- Data Generation ---
def generate_synthetic_data(periods=5000):
    """Generates synthetic 1-minute data with a clear M-pattern for testing."""
    rng = np.random.default_rng(42)

    index = pd.date_range(start='2023-01-01', periods=periods, freq='min')
    data = pd.DataFrame(index=index)

    price = 100 + np.cumsum(rng.normal(0, 0.1, periods))
    data['Open'] = price
    data['High'] = price + rng.uniform(0, 0.2, periods)
    data['Low'] = price - rng.uniform(0, 0.2, periods)
    data['Close'] = price + rng.normal(0, 0.05, periods)
    data['Volume'] = rng.integers(100, 1000, size=periods)

    # --- Inject a clear M-pattern (level drop) ---
    # A: Peak
    data.loc[data.index[1000:1010], ['High', 'Close']] += 5
    # B: Trough
    data.loc[data.index[1100:1110], ['Low', 'Close']] -= 5
    # C: Retracement Peak (to 50% of A-B drop)
    data.loc[data.index[1200:1210], ['High', 'Close']] += 2.5

    # Add a bearish engulfing pattern right after the retracement peak
    engulf_idx = 1211
    prev_idx = 1210

    # Make previous candle bullish
    data.iloc[prev_idx, data.columns.get_loc('Open')] = data.iloc[prev_idx, data.columns.get_loc('Close')] - 0.2

    # Make current candle a bearish engulfing one
    data.iloc[engulf_idx, data.columns.get_loc('Open')] = data.iloc[prev_idx, data.columns.get_loc('Close')] + 0.1
    data.iloc[engulf_idx, data.columns.get_loc('Close')] = data.iloc[prev_idx, data.columns.get_loc('Open')] - 0.1
    data.iloc[engulf_idx, data.columns.get_loc('High')] = data.iloc[engulf_idx, data.columns.get_loc('Open')] + 0.05
    data.iloc[engulf_idx, data.columns.get_loc('Low')] = data.iloc[engulf_idx, data.columns.get_loc('Close')] - 0.05

    # D: Continuation Down
    data.loc[data.index[1300:1310], ['Low', 'Close']] -= 7

    return data

# --- Pre-processing ---
def preprocess_data(df, peak_prominence=1.0):
    """
    Analyzes 15M data to find level drops and merges the signals back into the 1M data.
    """
    df_15m = df['Close'].resample('15min').ohlc()
    df_15m.columns = ['Open', 'High', 'Low', 'Close']

    high_peaks_idx, _ = find_peaks(df_15m['High'], prominence=peak_prominence)
    low_troughs_idx, _ = find_peaks(-df_15m['Low'], prominence=peak_prominence)

    df_15m['is_swing_high'] = False
    df_15m.iloc[high_peaks_idx, df_15m.columns.get_loc('is_swing_high')] = True
    df_15m['is_swing_low'] = False
    df_15m.iloc[low_troughs_idx, df_15m.columns.get_loc('is_swing_low')] = True

    df_15m['aoi'] = np.nan
    df_15m['level_high'] = np.nan
    df_15m['level_low'] = np.nan

    swing_highs = df_15m[df_15m['is_swing_high']]
    swing_lows = df_15m[df_15m['is_swing_low']]

    for high_idx, high_peak in swing_highs.iterrows():
        next_lows = swing_lows[swing_lows.index > high_idx]
        if not next_lows.empty:
            next_low = next_lows.iloc[0]
            level_high_price = high_peak['High']
            level_low_price = next_low['Low']
            aoi = level_low_price + (level_high_price - level_low_price) * 0.5
            df_15m.loc[next_low.name, 'aoi'] = aoi
            df_15m.loc[next_low.name, 'level_high'] = level_high_price
            df_15m.loc[next_low.name, 'level_low'] = level_low_price

    df_merged = pd.merge_asof(df, df_15m[['aoi', 'level_high', 'level_low']],
                            left_index=True, right_index=True, direction='backward')
    df_merged[['aoi', 'level_high', 'level_low']] = df_merged[['aoi', 'level_high', 'level_low']].ffill()

    return df_merged

# --- Strategy Class ---
class MeasuredRetracementScalpMowInternalStrategy(Strategy):
    ema_period = 50
    rr_ratio = 5
    sl_buffer_pct = 0.1

    def init(self):
        self.aoi = self.I(lambda x: x, self.data.df['aoi'], plot=True, name='AOI')
        self.ema = self.I(lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean(),
                          self.data.Close, self.ema_period)
        self.in_aoi_since = 0

    def next(self):
        if self.position:
            return

        current_aoi = self.aoi[-1]
        if np.isnan(current_aoi):
            self.in_aoi_since = 0
            return

        if self.in_aoi_since == 0 and self.data.High[-1] >= current_aoi:
            self.in_aoi_since = len(self.data)

        if self.in_aoi_since == 0:
            return

        if len(self.data) < 2: return

        is_prev_bullish = self.data.Close[-2] > self.data.Open[-2]
        is_curr_bearish = self.data.Close[-1] < self.data.Open[-1]
        is_engulfing = (self.data.Open[-1] >= self.data.Close[-2] and
                        self.data.Close[-1] < self.data.Open[-2])
        is_bearish_engulfing = is_prev_bullish and is_curr_bearish and is_engulfing

        is_below_ema = self.data.Close[-1] < self.ema[-1]

        if is_bearish_engulfing and is_below_ema:
            entry_price = self.data.Close[-1]
            sl_buffer = entry_price * (self.sl_buffer_pct / 100)
            sl = self.data.High[-1] + sl_buffer

            risk = abs(entry_price - sl)
            if risk <= 1e-9: return

            tp = entry_price - (risk * self.rr_ratio)
            if tp >= entry_price: return

            self.sell(sl=sl, tp=tp)
            self.in_aoi_since = 0

# --- Main Execution Block ---
if __name__ == '__main__':
    data_1m = generate_synthetic_data()

    best_stats = None
    best_prominence = None

    for prominence in np.arange(0.5, 2.5, 0.5):
        try:
            processed_data = preprocess_data(data_1m.copy(), peak_prominence=prominence)
            if processed_data.empty or processed_data['aoi'].isnull().all():
                continue

            bt = Backtest(processed_data, MeasuredRetracementScalpMowInternalStrategy, cash=100_000, commission=.002)
            stats = bt.optimize(ema_period=range(20, 100, 20),
                                rr_ratio=range(3, 8, 1),
                                maximize='Sharpe Ratio',
                                max_tries=50)

            bt = Backtest(processed_data, MeasuredRetracementScalpMowInternalStrategy, cash=100_000, commission=.002)
            stats = bt.optimize(ema_period=range(20, 100, 20),
                                rr_ratio=range(3, 8, 1),
                                maximize='Sharpe Ratio',
                                max_tries=50)

            if best_stats is None or stats.get('Sharpe Ratio', -1) > best_stats.get('Sharpe Ratio', -1):
                best_stats = stats
                best_prominence = prominence
                # A new best is found, so we generate the plot from the current Backtest instance
                try:
                    bt.plot(filename='results/measured_retracement_scalp.html', open_browser=False)
                except Exception as e:
                    print(f"Could not generate plot for prominence {prominence}: {e}")
        except Exception as e:
            print(f"Error during optimization with prominence {prominence}: {e}")
            continue

    if best_stats is None:
        raise ValueError("Optimization failed to find any valid results.")

    print(f"Best Prominence: {best_prominence}")
    print("--- Best Stats ---")
    print(best_stats)

    os.makedirs('results', exist_ok=True)

    results_dict = {
        'strategy_name': 'measured_retracement_scalp_mow_internal',
        'return': float(best_stats.get('Return [%]', 0.0)),
        'sharpe': float(best_stats.get('Sharpe Ratio', 0.0)),
        'max_drawdown': float(best_stats.get('Max. Drawdown [%]', 0.0)),
        'win_rate': float(best_stats.get('Win Rate [%]', 0.0)),
        'total_trades': int(best_stats.get('# Trades', 0))
    }
    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
