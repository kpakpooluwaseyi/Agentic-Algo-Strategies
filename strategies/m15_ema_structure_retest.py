
import json
import os
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from scipy.signal import find_peaks

# Custom EMA implementation to avoid external dependencies
def EMA(series, n):
    """Calculates the Exponential Moving Average."""
    return pd.Series(series).ewm(span=n, adjust=False).mean().values

def _find_swing_points(series, distance, is_high):
    """Helper function to find swing points (peaks/troughs) in a series."""
    if is_high:
        peaks, _ = find_peaks(series, distance=distance, prominence=0.01)
    else:
        # Invert series to find troughs as peaks
        peaks, _ = find_peaks(-series, distance=distance, prominence=0.01)

    signal = np.full(len(series), np.nan, dtype=float)
    signal[peaks] = series[peaks]
    return signal

class M15EmaStructureRetestStrategy(Strategy):
    ema_fast_period = 50
    ema_slow_period = 200
    swing_distance = 15
    retest_tolerance = 0.01 # 1% tolerance for retest

    def init(self):
        self.ema_fast = self.I(EMA, self.data.Close, self.ema_fast_period)
        self.ema_slow = self.I(EMA, self.data.Close, self.ema_slow_period)

        self.swing_highs = self.I(_find_swing_points, self.data.High, distance=self.swing_distance, is_high=True)
        self.swing_lows = self.I(_find_swing_points, self.data.Low, distance=self.swing_distance, is_high=False)

        # State machine variables
        self.ema_cross_direction = 0  # 1 for bullish, -1 for bearish
        self.structure_level = None
        self.impulse_start_level = None
        self.state = 'WAIT_CROSS' # States: WAIT_CROSS, WAIT_BREAK, WAIT_RETEST

    def _is_bullish_engulfing(self):
        if len(self.data.Close) < 2: return False
        prev_open, prev_close = self.data.Open[-2], self.data.Close[-2]
        curr_open, curr_close = self.data.Open[-1], self.data.Close[-1]
        return (prev_close < prev_open and curr_close > curr_open and
                curr_close >= prev_open and curr_open <= prev_close)

    def _is_bearish_engulfing(self):
        if len(self.data.Close) < 2: return False
        prev_open, prev_close = self.data.Open[-2], self.data.Close[-2]
        curr_open, curr_close = self.data.Open[-1], self.data.Close[-1]
        return (prev_close > prev_open and curr_close < curr_open and
                curr_close <= prev_open and curr_open >= prev_close)

    def _is_hammer(self):
        if len(self.data.Close) < 1: return False
        o, h, l, c = self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1]
        body = abs(c - o)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        return body > 0 and lower_wick > 2 * body and upper_wick < body

    def next(self):
        price = self.data.Close[-1]

        # --- State: WAIT_CROSS ---
        if self.state == 'WAIT_CROSS':
            if crossover(self.ema_fast, self.ema_slow):
                self.ema_cross_direction = 1
                self.state = 'WAIT_BREAK'
            elif crossover(self.ema_slow, self.ema_fast):
                self.ema_cross_direction = -1
                self.state = 'WAIT_BREAK'
            return

        # Reset if EMA cross invalidates the setup
        if (self.ema_cross_direction == 1 and self.ema_fast[-1] < self.ema_slow[-1]) or \
           (self.ema_cross_direction == -1 and self.ema_fast[-1] > self.ema_slow[-1]):
            self._reset_state()
            return

        # --- State: WAIT_BREAK ---
        if self.state == 'WAIT_BREAK':
            if self.ema_cross_direction == 1: # Bullish setup
                # Find the last significant high to act as structure
                recent_high_indices = np.where(~np.isnan(self.swing_highs))[0]
                if len(recent_high_indices) > 0:
                    last_high_idx = recent_high_indices[-1]
                    self.structure_level = self.data.High[last_high_idx]

                    # Find the low before that high
                    recent_low_indices = np.where((~np.isnan(self.swing_lows)) & (np.arange(len(self.data.Close)) < last_high_idx) )[0]
                    if len(recent_low_indices)>0:
                        self.impulse_start_level = self.data.Low[recent_low_indices[-1]]

                if self.structure_level and price > self.structure_level:
                    self.state = 'WAIT_RETEST'

            elif self.ema_cross_direction == -1: # Bearish setup
                recent_low_indices = np.where(~np.isnan(self.swing_lows))[0]
                if len(recent_low_indices) > 0:
                    last_low_idx = recent_low_indices[-1]
                    self.structure_level = self.data.Low[last_low_idx]

                    recent_high_indices = np.where((~np.isnan(self.swing_highs)) & (np.arange(len(self.data.Close)) < last_low_idx) )[0]
                    if len(recent_high_indices)>0:
                        self.impulse_start_level = self.data.High[recent_high_indices[-1]]

                if self.structure_level and price < self.structure_level:
                    self.state = 'WAIT_RETEST'

        # --- State: WAIT_RETEST ---
        elif self.state == 'WAIT_RETEST':
            if self.position: return
            is_retesting_ema = abs(price - self.ema_fast[-1]) / price < self.retest_tolerance

            if is_retesting_ema and self.impulse_start_level:
                if self.ema_cross_direction == 1 and (self._is_bullish_engulfing() or self._is_hammer()):
                    sl = self.impulse_start_level
                    risk = price - sl
                    if risk <= 0: return
                    tp = price + 1.5 * risk
                    self.buy(sl=sl, tp=tp)
                    self._reset_state()

                elif self.ema_cross_direction == -1 and self._is_bearish_engulfing():
                    sl = self.impulse_start_level
                    risk = sl - price
                    if risk <= 0: return
                    tp = price - 1.5 * risk
                    self.sell(sl=sl, tp=tp)
                    self._reset_state()

    def _reset_state(self):
        self.ema_cross_direction = 0
        self.structure_level = None
        self.impulse_start_level = None
        self.state = 'WAIT_CROSS'


def generate_synthetic_data():
    """Generates synthetic data that ideally produces a long entry setup."""
    n = 2000
    time = pd.date_range('2020-01-01', periods=n, freq='15min')
    price = 100
    prices = []

    # 1. Initial ranging/downtrend for EMA setup
    for _ in range(500):
        price += np.random.uniform(-0.1, 0.05)
        prices.append(price)

    # 2. Strong upward move to create bullish EMA cross
    for _ in range(200):
        price += np.random.uniform(0.1, 0.2)
        prices.append(price)

    # 3. Impulse 1 (forms a peak)
    for _ in range(100):
        price += np.random.uniform(0.05, 0.15)
        prices.append(price)
    structure_high = price

    # 4. Pullback
    for _ in range(100):
        price -= np.random.uniform(0.05, 0.1)
        prices.append(price)
    structure_low = price

    # 5. Impulse 2 (breaks structure)
    for _ in range(150):
        price += np.random.uniform(0.1, 0.2)
        if price > structure_high: # Ensure break
            price += 0.1
        prices.append(price)

    # 6. Pullback to EMA for retest
    for _ in range(150):
        price -= np.random.uniform(0.08, 0.15)
        prices.append(price)

    # 7. Bullish engulfing candle at retest
    prices.append(price)      # Prev Close (lower)
    prices.append(price+0.1)  # Prev Open (higher) -> Bearish prev candle
    prices.append(price-0.05) # Curr Open (lower than prev close)
    prices.append(price+0.2)  # Curr Close (higher than prev open) -> Bullish engulfing

    # Fill remaining data
    while len(prices) < n:
        price += np.random.uniform(-0.1, 0.12)
        prices.append(price)

    df = pd.DataFrame(index=time)
    df['Open'] = prices
    df['High'] = df['Open'] + np.random.uniform(0.01, 0.1, n)
    df['Low'] = df['Open'] - np.random.uniform(0.01, 0.1, n)
    df['Close'] = df['Open'] + np.random.uniform(-0.05, 0.05, n)
    df['Volume'] = np.random.randint(100, 1000, n)
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0.01, 0.05, n)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0.01, 0.05, n)
    return df

def sanitize_for_json(obj):
    """Recursively sanitize an object for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
        return None  # Cannot serialize these types
    elif pd.isna(obj):
        return None
    return obj


if __name__ == '__main__':
    data = generate_synthetic_data()

    bt = Backtest(data, M15EmaStructureRetestStrategy, cash=100_000, commission=.002)

    stats = bt.optimize(
        ema_fast_period=range(40, 61, 10),
        ema_slow_period=range(180, 221, 20),
        swing_distance=range(10, 21, 5),
        retest_tolerance=[i/1000 for i in range(5, 21, 5)],
        maximize='Sharpe Ratio',
        constraint=lambda p: p.ema_fast_period < p.ema_slow_period
    )

    print("Best Run Stats:")
    print(stats)

    results = sanitize_for_json(stats.to_dict())

    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'm15_ema_structure_retest',
            'return': results.get('Return [%]'),
            'sharpe': results.get('Sharpe Ratio'),
            'max_drawdown': results.get('Max. Drawdown [%]'),
            'win_rate': results.get('Win Rate [%]'),
            'total_trades': results.get('# Trades')
        }, f, indent=2)

    print("\nResults saved to results/temp_result.json")

    try:
        bt.plot(filename="results/m15_ema_structure_retest", open_browser=False)
        print("Plot saved to results/m15_ema_structure_retest.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
