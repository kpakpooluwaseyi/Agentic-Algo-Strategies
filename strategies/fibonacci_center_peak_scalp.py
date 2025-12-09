import json
import os
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# --- Synthetic Data Generation ---
def generate_synthetic_data(num_candles=1000):
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_candles, freq='15min'))
    df = pd.DataFrame(index=dates)
    price = 100 + 5 * np.sin(np.linspace(0, 20, num_candles)) + np.random.randn(num_candles) * 0.5

    p1_idx, p1_high = 100, 115
    p2_idx, p2_low = 150, 95
    p3_idx = 200
    fib_50_level = p2_low + (p1_high - p2_low) * 0.5
    p3_peak = fib_50_level + 1

    # Make the pattern cleaner by removing noise in the key areas
    price[p1_idx:p2_idx+1] = np.linspace(p1_high, p2_low, p2_idx - p1_idx + 1)
    price[p2_idx+1:p3_idx+1] = np.linspace(p2_low, p3_peak, p3_idx - p2_idx)
    price[p3_idx+3:p3_idx+50] = np.linspace(p3_peak - 2.5, p2_low - 5, 47)

    df['Open'] = price
    df['High'] = df['Open'] + np.random.uniform(0, 0.2, num_candles)
    df['Low'] = df['Open'] - np.random.uniform(0, 0.2, num_candles)
    df['Close'] = df['Open'] + np.random.uniform(-0.1, 0.1, num_candles)
    df['Volume'] = np.random.randint(100, 1000, num_candles)

    df.at[df.index[p1_idx], 'High'] = p1_high
    df.at[df.index[p2_idx], 'Low'] = p2_low
    df.at[df.index[p3_idx], 'High'] = p3_peak
    df.at[df.index[p3_idx], 'Close'], df.at[df.index[p3_idx], 'Open'] = p3_peak - 0.1, p3_peak - 0.6
    df.at[df.index[p3_idx+1], 'Open'], df.at[df.index[p3_idx+1], 'High'] = p3_peak, p3_peak + 0.2
    df.at[df.index[p3_idx+1], 'Close'], df.at[df.index[p3_idx+1], 'Low'] = df.at[df.index[p3_idx], 'Open'] - 0.1, df.at[df.index[p3_idx], 'Open'] - 0.2

    return df

def find_swing_points(series, lookback, find_highs=True):
    series_np = np.asarray(series)
    if find_highs: peaks, _ = find_peaks(series_np, distance=lookback)
    else: peaks, _ = find_peaks(-series_np, distance=lookback)
    swings = np.zeros_like(series_np, dtype=bool)
    swings[peaks] = True
    return swings

class FibonacciCenterPeakScalpStrategy(Strategy):
    swing_lookback = 30 # Increased default to avoid noise
    min_rr = 1.5
    sl_buffer_pct = 0.01

    def init(self):
        self.reset_state()
        self.last_processed_high_idx = -1
        self.swing_highs = self.I(find_swing_points, self.data.High, self.swing_lookback, True)
        self.swing_lows = self.I(find_swing_points, self.data.Low, self.swing_lookback, False)

    def next(self):
        current_index = len(self.data) - 1
        if self.position: return

        if self.state == "SEARCHING_INITIAL_DROP":
            high_indices = np.where(self.swing_highs)[0]
            if not len(high_indices): return

            latest_high_idx = high_indices[-1]
            if latest_high_idx <= self.last_processed_high_idx or latest_high_idx >= current_index: return

            low_indices_after_high = np.where(self.swing_lows[latest_high_idx:current_index])[0]
            if not len(low_indices_after_high): return

            p2_idx = latest_high_idx + low_indices_after_high[0]
            p1_idx = latest_high_idx

            self.point1_high, self.point1_high_idx = self.data.High[p1_idx], p1_idx
            self.point2_low, self.point2_low_idx = self.data.Low[p2_idx], p2_idx

            drop_range = self.point1_high - self.point2_low
            if drop_range > 0:
                self.fib_50_level = self.point2_low + (drop_range * 0.5)
                self.last_processed_high_idx = p1_idx
                self.retracement_high = self.point2_low
                self.state = "WAITING_FOR_RETRACEMENT"

        elif self.state == "WAITING_FOR_RETRACEMENT":
            self.retracement_high = max(self.retracement_high, self.data.High[-1])
            if self.data.Close[-1] < self.point2_low: self.reset_state()
            elif self.data.High[-1] >= self.fib_50_level: self.state = "LOOKING_FOR_REVERSAL"

        elif self.state == "LOOKING_FOR_REVERSAL":
            self.retracement_high = max(self.retracement_high, self.data.High[-1])
            if self.data.Close[-1] < self.point2_low or self.data.Close[-1] > self.point1_high:
                self.reset_state()
                return

            if len(self.data.Close) < 2: return
            is_prev_bullish = self.data.Close[-2] > self.data.Open[-2]
            is_curr_bearish = self.data.Close[-1] < self.data.Open[-1]
            is_engulfing = self.data.Open[-1] > self.data.Close[-2] and self.data.Close[-1] < self.data.Open[-2]

            if is_prev_bullish and is_curr_bearish and is_engulfing:
                center_peak_high = self.retracement_high
                entry_price = self.data.Close[-1]
                stop_loss = center_peak_high * (1 + self.sl_buffer_pct)
                take_profit = center_peak_high - (center_peak_high - self.point2_low) * 0.5

                reward = entry_price - take_profit
                risk = stop_loss - entry_price

                if risk > 0 and reward / risk >= self.min_rr:
                    self.sell(sl=stop_loss, tp=take_profit)
                self.reset_state()

    def reset_state(self):
        self.point1_high, self.point1_high_idx = None, None
        self.point2_low, self.point2_low_idx = None, None
        self.fib_50_level = None
        self.retracement_high = None
        self.state = "SEARCHING_INITIAL_DROP"

if __name__ == '__main__':
    data = generate_synthetic_data(num_candles=500)
    bt = Backtest(data, FibonacciCenterPeakScalpStrategy, cash=100_000, commission=.002)

    print("Running single backtest with default parameters...")
    stats = bt.run()
    print(stats)

    print("Optimizing strategy...")
    stats_opt = bt.optimize(
        swing_lookback=range(10, 40, 10),
        min_rr=[1.0, 1.5, 2.0],
        sl_buffer_pct=list(np.arange(0.005, 0.02, 0.005)),
        maximize='Sharpe Ratio'
    )

    print("Best run stats (from optimization):")
    print(stats_opt)

    print("Saving results...")
    os.makedirs('results', exist_ok=True)

    def sanitize_for_json(obj):
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    final_stats = stats_opt if stats_opt is not None else stats
    results_dict = {
        'strategy_name': 'fibonacci_center_peak_scalp',
        'return': final_stats.get('Return [%]', 0.0),
        'sharpe': final_stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': final_stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': final_stats.get('Win Rate [%]', 0.0),
        'total_trades': final_stats.get('# Trades', 0)
    }
    results_dict = {k: sanitize_for_json(v) for k, v in results_dict.items()}

    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("Generating plot...")
    try:
        bt.plot(filename="results/fibonacci_center_peak_scalp.html", open_browser=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")

    print("Script finished successfully.")
