from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os
from scipy.signal import find_peaks

def generate_synthetic_data(periods=1000):
    """
    Generates synthetic 1-minute OHLC data that exhibits the specific
    Fibonacci Center Peak Scalp pattern.
    """
    np.random.seed(42)
    index = pd.date_range(start='2023-01-01', periods=periods, freq='T')

    price = np.zeros(periods)

    # Phase 1: Initial Drop (High to Low)
    start_price = 100
    low_price = 90
    drop_duration = 200
    price[:drop_duration] = np.linspace(start_price, low_price, drop_duration)

    # Phase 2: Retracement to 50% Fib level
    high_price = start_price
    fib_50 = low_price + (high_price - low_price) * 0.5
    retracement_duration = 150
    retracement_end = drop_duration + retracement_duration
    price[drop_duration:retracement_end] = np.linspace(low_price, fib_50, retracement_duration)

    # Phase 3: Second Drop
    second_drop_target = 85
    second_drop_duration = 250
    second_drop_end = retracement_end + second_drop_duration
    price[retracement_end:second_drop_end] = np.linspace(fib_50, second_drop_target, second_drop_duration)

    # Phase 4: Random walk for the rest
    remaining_periods = periods - second_drop_end
    if remaining_periods > 0:
        random_walk = np.random.randn(remaining_periods).cumsum() * 0.1
        price[second_drop_end:] = second_drop_target + random_walk

    price += np.random.randn(periods) * 0.05

    df = pd.DataFrame(index=index)
    df['Open'] = price
    df['High'] = df['Open'] + np.random.uniform(0, 0.1, periods)
    df['Low'] = df['Open'] - np.random.uniform(0, 0.1, periods)
    df['Close'] = df['Open'] + np.random.uniform(-0.05, 0.05, periods)

    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 0.05, periods)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 0.05, periods)

    # Explicitly set key points
    df.iloc[0, df.columns.get_loc('High')] = start_price + 0.1
    df.iloc[drop_duration - 1, df.columns.get_loc('Low')] = low_price - 0.1
    df.iloc[retracement_end - 1, df.columns.get_loc('High')] = fib_50 + 0.2 # Stop hunt wick

    return df

def preprocess_data(df_1m, prominence=1, peak_distance=3):
    """
    Simulates 15M analysis on 1M data to find Fibonacci setups.
    - Resamples to 15M to find major swing points.
    - Calculates entry AOI, SL, and TP levels.
    - Merges this analysis back into the 1M dataframe.
    """
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()

    highs, _ = find_peaks(df_15m['High'], prominence=prominence, distance=peak_distance)
    lows, _ = find_peaks(-df_15m['Low'], prominence=prominence, distance=peak_distance)

    df_1m['signal'] = 0
    df_1m['aoi_high'] = np.nan
    df_1m['aoi_low'] = np.nan
    df_1m['stop_loss'] = np.nan
    df_1m['take_profit'] = np.nan

    for i in range(len(highs) - 1):
        for j in range(len(lows)):
            if lows[j] > highs[i]:
                initial_high_idx = highs[i]
                initial_low_idx = lows[j]

                initial_high_price = df_15m.iloc[initial_high_idx]['High']
                initial_low_price = df_15m.iloc[initial_low_idx]['Low']

                fib_50 = initial_low_price + (initial_high_price - initial_low_price) * 0.5
                fib_382 = initial_low_price + (initial_high_price - initial_low_price) * 0.382
                fib_618 = initial_low_price + (initial_high_price - initial_low_price) * 0.618

                # Define a realistic window to search for the center peak
                search_window_start = initial_low_idx
                search_window_end = lows[j] + 20 # Look for ~5 hours (20 * 15min)
                retracement_df = df_15m.iloc[search_window_start:search_window_end]

                center_peak_candidates = retracement_df[(retracement_df['High'] > fib_382) & (retracement_df['High'] < fib_618)]

                if not center_peak_candidates.empty:
                    center_peak_high = center_peak_candidates['High'].max()
                    center_peak_time = center_peak_candidates['High'].idxmax()

                    if abs(center_peak_high - fib_50) < (fib_50 - fib_382):

                        sl = center_peak_high + 0.1
                        tp = center_peak_high - (center_peak_high - initial_low_price) * 0.5

                        start_time = df_15m.index[initial_low_idx]
                        # The signal is valid until a new major low is formed
                        next_low_candidates = [l for l in lows if l > initial_low_idx]
                        end_time = df_15m.index[next_low_candidates[0]] if next_low_candidates else df_15m.index[-1]

                        aoi_buffer = (fib_50 - fib_382) * 0.25 # Tighter AOI

                        df_1m.loc[start_time:end_time, 'signal'] = 1
                        df_1m.loc[start_time:end_time, 'aoi_high'] = fib_50 + aoi_buffer
                        df_1m.loc[start_time:end_time, 'aoi_low'] = fib_50 - aoi_buffer
                        df_1m.loc[start_time:end_time, 'stop_loss'] = sl
                        df_1m.loc[start_time:end_time, 'take_profit'] = tp

    df_1m[['aoi_high', 'aoi_low', 'stop_loss', 'take_profit']] = df_1m[['aoi_high', 'aoi_low', 'stop_loss', 'take_profit']].ffill()

    return df_1m

class FibonacciCenterPeakScalpStrategy(Strategy):
    min_rr = 5
    confirmation_lookback = 3

    def init(self):
        self.signal = self.I(lambda x: x, self.data.df['signal'], name='signal')
        self.aoi_low = self.I(lambda x: x, self.data.df['aoi_low'], name='aoi_low')
        self.aoi_high = self.I(lambda x: x, self.data.df['aoi_high'], name='aoi_high')
        self.stop_loss = self.I(lambda x: x, self.data.df['stop_loss'], name='stop_loss')
        self.take_profit = self.I(lambda x: x, self.data.df['take_profit'], name='take_profit')

    def next(self):
        if self.signal[-1] == 1 and not self.position:

            price = self.data.Close[-1]

            if self.aoi_low[-1] < price < self.aoi_high[-1]:

                is_bearish_confirmation = False
                for i in range(1, self.confirmation_lookback + 1):
                    if len(self.data.Close) > i and self.data.Close[-i] < self.data.Open[-i]:
                        is_bearish_confirmation = True
                        break

                if is_bearish_confirmation:
                    sl = self.stop_loss[-1]
                    tp = self.take_profit[-1]

                    if sl > price and tp < price:
                        rr = abs(price - tp) / abs(price - sl)
                        if rr >= self.min_rr:
                            self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    # Generate synthetic data
    data = generate_synthetic_data(periods=5000)

    # Preprocess data to add signals
    # Adjusted prominence and distance to be less strict for synthetic data
    processed_data = preprocess_data(data, prominence=1, peak_distance=5)

    # Run backtest
    bt = Backtest(processed_data, FibonacciCenterPeakScalpStrategy, cash=100000, commission=.002)

    # Optimize
    stats = bt.optimize(
        min_rr=range(3, 8, 1),
        confirmation_lookback=range(2, 6, 1),
        maximize='Sharpe Ratio'
    )

    # Save results
    os.makedirs('results', exist_ok=True)

    # Handle potential NaN values from stats
    sharpe_ratio = stats.get('Sharpe Ratio', 0)
    win_rate = stats.get('Win Rate [%]', 0)

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'fibonacci_center_peak_scalp',
            'return': float(stats.get('Return [%]', 0)),
            'sharpe': float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0,
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)),
            'win_rate': float(win_rate) if not np.isnan(win_rate) else 0,
            'total_trades': int(stats.get('# Trades', 0))
        }, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    # Generate plot
    try:
        bt.plot(filename='results/fibonacci_center_peak_scalp.html')
        print("Backtest plot saved to results/fibonacci_center_peak_scalp.html")
    except Exception as e:
        print(f"Could not generate plot due to: {e}")
