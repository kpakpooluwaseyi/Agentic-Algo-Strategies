
import json
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import os

# --- Helper Functions ---

def EMA(series, n):
    """Calculates the Exponential Moving Average."""
    return pd.Series(series).ewm(span=n, adjust=False).mean().values

def find_swing_points(highs, lows, window=50):
    """
    Finds the most recent completed swing high and low.
    Returns (swing_high, swing_low, high_idx, low_idx)
    """
    if len(highs) < window:
        return None, None, None, None

    window_highs = highs[-window:]
    window_lows = lows[-window:]

    high_idx_window = np.argmax(window_highs)
    low_idx_window = np.argmin(window_lows)

    swing_high = window_highs[high_idx_window]
    swing_low = window_lows[low_idx_window]

    high_idx_abs = len(highs) - window + high_idx_window
    low_idx_abs = len(lows) - window + low_idx_window

    return swing_high, swing_low, high_idx_abs, low_idx_abs

def sanitize_stats(stats):
    """Converts numpy types and NaN to native Python types for JSON."""
    # Handle complex pandas objects first
    if isinstance(stats, (pd.DataFrame, pd.Series)):
        return None

    if isinstance(stats, dict):
        return {k: sanitize_stats(v) for k, v in stats.items()}
    elif isinstance(stats, (list, tuple)):
        return [sanitize_stats(i) for i in stats]

    if pd.isna(stats):
        return None

    if isinstance(stats, np.integer):
        return int(stats)
    elif isinstance(stats, np.floating):
        return float(stats)
    elif isinstance(stats, np.ndarray):
        return stats.tolist()

    return stats

# --- Data Generation ---
def generate_synthetic_data(periods=2000):
    """Generates synthetic 1-minute OHLCV data for the strategy."""
    np.random.seed(42)
    initial_price = 100
    price = initial_price - np.logspace(0, 1.5, num=300)
    level_high, level_low = initial_price, price[-1]
    fib_50_level = level_low + (level_high - level_low) * 0.5
    price = np.append(price, np.linspace(level_low, fib_50_level, num=200))
    price = np.append(price, price[-1] + 0.1)
    price = np.append(price, price[-2] - 0.2)
    price = np.append(price, price[-1] + np.random.randn(50 - 2) * 0.05)
    retrace_high = price[500]
    measured_move_target = retrace_high - (retrace_high - level_low) * 0.5
    price = np.append(price, np.linspace(price[-1], measured_move_target, num=300))
    price = np.append(price, price[-1] + np.random.randn(periods - len(price)) * 0.1)
    index = pd.date_range(start='2023-01-01', periods=periods, freq='1min')
    df = pd.DataFrame(index=index)
    df['Close'] = price
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.rand(periods) * 0.05
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.rand(periods) * 0.05
    df['Volume'] = np.random.randint(100, 1000, size=periods)
    df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    return df.iloc[51:]

# --- Strategy Class ---
class Fibonacci5050MeasuredMoveScalpStrategy(Strategy):
    risk_reward_ratio = 2.0
    aoi_zone_percent = 0.01
    swing_window = 100
    ema_period = 200

    def init(self):
        self.ema = self.I(EMA, self.data.Close, self.ema_period)

    def next(self):
        if self.position or len(self.data.High) < self.swing_window:
            return

        price = self.data.Close[-1]

        swing_high, swing_low, high_idx, low_idx = find_swing_points(self.data.High, self.data.Low, self.swing_window)
        if high_idx is None: return

        if high_idx < low_idx:
            aoi_short = swing_low + (swing_high - swing_low) * 0.5
            if price > aoi_short * (1 - self.aoi_zone_percent) and price < self.ema[-1]:
                if (self.data.Close[-1] < self.data.Open[-1] and self.data.Open[-1] >= self.data.Close[-2] and self.data.Close[-1] < self.data.Open[-2] and abs(self.data.Close[-1] - self.data.Open[-1]) > abs(self.data.Close[-2] - self.data.Open[-2])):
                    retrace_peak = self.data.High[low_idx:].max()
                    sl = retrace_peak * 1.001
                    tp = retrace_peak - (retrace_peak - swing_low) * 0.5
                    risk, reward = abs(sl - price), abs(price - tp)
                    if risk > 0 and reward / risk >= self.risk_reward_ratio and tp < price:
                        self.sell(sl=sl, tp=tp)

        elif low_idx < high_idx:
            aoi_long = swing_high - (swing_high - swing_low) * 0.5
            if price < aoi_long * (1 + self.aoi_zone_percent) and price > self.ema[-1]:
                if (self.data.Close[-1] > self.data.Open[-1] and self.data.Open[-1] <= self.data.Close[-2] and self.data.Close[-1] > self.data.Open[-2] and abs(self.data.Close[-1] - self.data.Open[-1]) > abs(self.data.Close[-2] - self.data.Open[-2])):
                    retrace_trough = self.data.Low[high_idx:].min()
                    sl = retrace_trough * 0.999
                    tp = retrace_trough + (swing_high - retrace_trough) * 0.5
                    risk, reward = abs(price - sl), abs(tp - price)
                    if risk > 0 and reward / risk >= self.risk_reward_ratio and tp > price:
                        self.buy(sl=sl, tp=tp)

# --- Main Execution Block ---
if __name__ == '__main__':
    data = generate_synthetic_data()
    bt = Backtest(data, Fibonacci5050MeasuredMoveScalpStrategy, cash=100_000, commission=.002)

    print("Optimizing strategy...")
    stats = bt.optimize(
        risk_reward_ratio=np.arange(1.5, 5.5, 0.5).tolist(),
        swing_window=range(50, 150, 25),
        maximize='Sharpe Ratio'
    )

    print("\n--- Best Optimization Stats ---")
    print(stats)

    os.makedirs('results', exist_ok=True)
    stats_dict = stats.to_dict()
    sanitized_for_json = sanitize_stats(stats_dict)

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'fibonacci_50_50_measured_move_scalp',
            'return': sanitized_for_json.get('Return [%]'),
            'sharpe': sanitized_for_json.get('Sharpe Ratio'),
            'max_drawdown': sanitized_for_json.get('Max. Drawdown [%]'),
            'win_rate': sanitized_for_json.get('Win Rate [%]'),
            'total_trades': sanitized_for_json.get('# Trades')
        }, f, indent=2)

    print("\nResults saved to results/temp_result.json")

    print("Generating plot...")
    try:
        bt.plot(filename="results/fibonacci_50_50_scalp_plot.html")
        print("Plot saved to results/fibonacci_50_50_scalp_plot.html")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
