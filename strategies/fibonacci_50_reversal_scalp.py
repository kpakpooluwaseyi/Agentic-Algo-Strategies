from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks
from enum import Enum

# --- State Machine Enum ---
class TradeState(Enum):
    SEARCHING = 0
    WAITING_FOR_CONFIRMATION_SHORT = 1
    WAITING_FOR_CONFIRMATION_LONG = 2

# --- Data Generation & Pre-processing ---
def generate_synthetic_data(num_points=5000):
    np.random.seed(42)
    time = np.arange(num_points)
    close = 100 + 0.01 * time + np.random.randn(num_points).cumsum()

    for _ in range(10): # M-patterns
        idx = np.random.randint(200, num_points - 200)
        start_price = close[idx]
        close[idx:idx+50] = np.linspace(start_price, start_price - 10, 50)
        close[idx+50:idx+100] = np.linspace(start_price - 10, start_price - 5, 50)
        close[idx+100:idx+150] = np.linspace(start_price - 5, start_price - 12, 50)

    for _ in range(10): # W-patterns
        idx = np.random.randint(200, num_points - 200)
        start_price = close[idx]
        close[idx:idx+50] = np.linspace(start_price, start_price + 10, 50)
        close[idx+50:idx+100] = np.linspace(start_price + 10, start_price + 5, 50)
        close[idx+100:idx+150] = np.linspace(start_price + 5, start_price + 12, 50)

    open_price = close + np.random.uniform(-0.1, 0.1, num_points)
    high = np.maximum(open_price, close) + np.random.uniform(0, 0.2, num_points)
    low = np.minimum(open_price, close) - np.random.uniform(0, 0.2, num_points)

    return pd.DataFrame({
        'Open': open_price, 'High': high, 'Low': low, 'Close': close
    }, index=pd.date_range(start='2023-01-01', periods=num_points, freq='min'))

def preprocess_data(data_1m, prominence=1):
    data_15m = data_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()

    peaks, _ = find_peaks(data_15m['High'], prominence=prominence)
    troughs, _ = find_peaks(-data_15m['Low'], prominence=prominence)

    data_15m['swing_high'] = 0
    data_15m['swing_low'] = 0
    data_15m.iloc[peaks, data_15m.columns.get_loc('swing_high')] = 1
    data_15m.iloc[troughs, data_15m.columns.get_loc('swing_low')] = 1

    swings = data_15m[(data_15m['swing_high'] == 1) | (data_15m['swing_low'] == 1)].copy()
    swings['price'] = np.where(swings['swing_high'] == 1, swings['High'], swings['Low'])

    data_15m.loc[:, ['level1_high', 'level1_low', 'fib_50_short_aoi', 'fib_50_long_aoi']] = np.nan

    last_swing = None
    for i in range(len(swings)):
        current_swing = swings.iloc[i]
        if last_swing is not None:
            if current_swing['swing_low'] and last_swing['swing_high']:
                fib_level = current_swing['price'] + (last_swing['price'] - current_swing['price']) * 0.5
                data_15m.loc[current_swing.name:, ['level1_high', 'level1_low', 'fib_50_short_aoi', 'fib_50_long_aoi']] = [last_swing['price'], current_swing['price'], fib_level, np.nan]
            elif current_swing['swing_high'] and last_swing['swing_low']:
                fib_level = current_swing['price'] - (current_swing['price'] - last_swing['price']) * 0.5
                data_15m.loc[current_swing.name:, ['level1_high', 'level1_low', 'fib_50_long_aoi', 'fib_50_short_aoi']] = [current_swing['price'], last_swing['price'], fib_level, np.nan]
        last_swing = current_swing

    data_1m = data_1m.merge(data_15m[['level1_high', 'level1_low', 'fib_50_short_aoi', 'fib_50_long_aoi']],
                            left_index=True, right_index=True, how='left').ffill()
    data_1m.dropna(inplace=True)
    return data_1m

# --- Strategy Class ---
class Fibonacci50ReversalScalpStrategy(Strategy):
    rr_ratio = 5.0
    aoi_tolerance = 0.002

    def init(self):
        self.state = TradeState.SEARCHING
        self.active_level1_high = None
        self.active_level1_low = None

    def is_bearish_engulfing(self):
        if len(self.data.Open) < 2: return False
        prev_candle_is_bullish = self.data.Close[-2] > self.data.Open[-2]
        curr_candle_is_bearish = self.data.Close[-1] < self.data.Open[-1]
        engulfs = self.data.Open[-1] >= self.data.Close[-2] and self.data.Close[-1] < self.data.Open[-2]
        return prev_candle_is_bullish and curr_candle_is_bearish and engulfs

    def is_bullish_engulfing(self):
        if len(self.data.Open) < 2: return False
        prev_candle_is_bearish = self.data.Close[-2] < self.data.Open[-2]
        curr_candle_is_bullish = self.data.Close[-1] > self.data.Open[-1]
        engulfs = self.data.Open[-1] <= self.data.Close[-2] and self.data.Close[-1] > self.data.Open[-2]
        return prev_candle_is_bearish and curr_candle_is_bullish and engulfs

    def next(self):
        if self.position:
            return

        current_price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        fib_short_aoi = self.data.fib_50_short_aoi[-1]
        fib_long_aoi = self.data.fib_50_long_aoi[-1]
        level1_high = self.data.level1_high[-1]
        level1_low = self.data.level1_low[-1]

        if self.state == TradeState.SEARCHING:
            if not np.isnan(fib_short_aoi) and high >= fib_short_aoi * (1 - self.aoi_tolerance):
                self.state = TradeState.WAITING_FOR_CONFIRMATION_SHORT
                self.active_level1_high = level1_high
                self.active_level1_low = level1_low
            elif not np.isnan(fib_long_aoi) and low <= fib_long_aoi * (1 + self.aoi_tolerance):
                self.state = TradeState.WAITING_FOR_CONFIRMATION_LONG
                self.active_level1_high = level1_high
                self.active_level1_low = level1_low

        elif self.state == TradeState.WAITING_FOR_CONFIRMATION_SHORT:
            if self.is_bearish_engulfing():
                center_peak = max(self.data.High[-2:])
                stop_loss = center_peak * 1.001
                take_profit = center_peak - (center_peak - self.active_level1_low) * 0.5

                if take_profit < current_price: # Validation
                    risk = abs(current_price - stop_loss)
                    reward = abs(current_price - take_profit)
                    if risk > 0 and reward / risk >= self.rr_ratio:
                        self.sell(sl=stop_loss, tp=take_profit)
                self.state = TradeState.SEARCHING
            elif high > self.active_level1_high:
                self.state = TradeState.SEARCHING

        elif self.state == TradeState.WAITING_FOR_CONFIRMATION_LONG:
            if self.is_bullish_engulfing():
                center_trough = min(self.data.Low[-2:])
                stop_loss = center_trough * 0.999
                take_profit = center_trough + (self.active_level1_high - center_trough) * 0.5

                if take_profit > current_price: # Validation
                    risk = abs(current_price - stop_loss)
                    reward = abs(current_price - take_profit)
                    if risk > 0 and reward / risk >= self.rr_ratio:
                        self.buy(sl=stop_loss, tp=take_profit)
                self.state = TradeState.SEARCHING
            elif low < self.active_level1_low:
                self.state = TradeState.SEARCHING

# --- Main Execution Block ---
if __name__ == '__main__':
    data = generate_synthetic_data(num_points=10000)

    best_stats = None
    best_prominence = 1
    for prominence in range(1, 11):
        try:
            processed_data = preprocess_data(data.copy(), prominence=prominence)
            if processed_data.empty: continue

            bt = Backtest(processed_data, Fibonacci50ReversalScalpStrategy, cash=10000, commission=.002)
            stats = bt.run()

            if stats['# Trades'] > 0:
                if best_stats is None or stats['Sharpe Ratio'] > best_stats.get('Sharpe Ratio', -np.inf):
                    best_stats = stats
                    best_prominence = prominence
        except Exception:
            continue

    if best_stats is None:
        print("No profitable strategy found.")
    else:
        print("Best Prominence:", best_prominence)
        print(best_stats)

        processed_data = preprocess_data(data.copy(), prominence=best_prominence)
        bt = Backtest(processed_data, Fibonacci50ReversalScalpStrategy, cash=10000, commission=.002)
        final_stats = bt.run()

        import os
        os.makedirs('results', exist_ok=True)
        with open('results/temp_result.json', 'w') as f:
            json.dump({
                'strategy_name': 'fibonacci_50_reversal_scalp',
                'return': final_stats.get('Return [%]', 0),
                'sharpe': final_stats.get('Sharpe Ratio', 0),
                'max_drawdown': final_stats.get('Max. Drawdown [%]', 0),
                'win_rate': final_stats.get('Win Rate [%]', 0),
                'total_trades': final_stats.get('# Trades', 0)
            }, f, indent=2)

        bt.plot(filename='results/fibonacci_50_reversal_scalp.html')
