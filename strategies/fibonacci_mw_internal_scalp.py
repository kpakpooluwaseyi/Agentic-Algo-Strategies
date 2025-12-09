import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from backtesting import Backtest, Strategy
import os

def generate_synthetic_data(periods=5000):
    """Generates synthetic 1M OHLC data with M/W patterns."""
    rng = np.random.default_rng(seed=42)
    price = 100 + np.cumsum(rng.normal(0, 0.01, periods))
    time = np.linspace(0, 10 * np.pi, periods)
    sine_wave = 2 * (np.sin(time) + np.sin(0.5 * time))
    price += sine_wave
    index = pd.date_range(start='2023-01-01', periods=periods, freq='min')
    df = pd.DataFrame(index=index)
    df['Open'] = price
    df['High'] = df['Open'] + rng.uniform(0, 0.1, periods)
    df['Low'] = df['Open'] - rng.uniform(0, 0.1, periods)
    df['Close'] = df['Open'] + rng.normal(0, 0.05, periods)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    return df

def preprocess_data(df_1m, prominence=1.0):
    """Pre-processes 1M data to identify 15M setups and AOIs."""
    df_15m = df_1m.resample('15min').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    peaks, _ = find_peaks(df_15m['High'], prominence=prominence)
    troughs, _ = find_peaks(-df_15m['Low'], prominence=prominence)
    peak_times = df_15m.index[peaks]
    trough_times = df_15m.index[troughs]
    swing_points = pd.concat([
        pd.DataFrame({'time': peak_times, 'type': 'peak', 'price': df_15m['High'].iloc[peaks]}),
        pd.DataFrame({'time': trough_times, 'type': 'trough', 'price': df_15m['Low'].iloc[troughs]})
    ]).sort_values(by='time').reset_index(drop=True)
    setups = []
    for i in range(1, len(swing_points)):
        prev_point = swing_points.iloc[i-1]
        curr_point = swing_points.iloc[i]

        move_range = abs(curr_point['price'] - prev_point['price'])

        if prev_point['type'] == 'peak' and curr_point['type'] == 'trough':
            setup_high = prev_point['price']
            setup_low = curr_point['price']
            setups.append({
                'start_time': prev_point['time'], 'end_time': curr_point['time'], 'setup_type': -1,
                'fib_500': setup_low + 0.500 * move_range,
                'setup_high': setup_high, 'setup_low': setup_low
            })
        elif prev_point['type'] == 'trough' and curr_point['type'] == 'peak':
            setup_low = prev_point['price']
            setup_high = curr_point['price']
            setups.append({
                'start_time': prev_point['time'], 'end_time': curr_point['time'], 'setup_type': 1,
                'fib_500': setup_high - 0.500 * move_range,
                'setup_high': setup_high, 'setup_low': setup_low
            })

    setup_df = pd.DataFrame(setups)
    if not setup_df.empty:
        setup_df.set_index('end_time', inplace=True)
        setup_df = setup_df.reindex(df_1m.index, method='ffill')
        df_1m = df_1m.join(setup_df.drop(columns='start_time'))

    df_1m = df_1m.shift(1)
    df_1m.dropna(inplace=True)
    return df_1m

def map_to_indicator(data):
    return data

class FibonacciMwInternalScalpStrategy(Strategy):
    sl_buffer_multiplier = 1.5
    sl_lookback = 5

    def init(self):
        self.setup_type = self.I(map_to_indicator, self.data.df['setup_type'].values, name="setup_type")
        self.fib_500 = self.I(map_to_indicator, self.data.df['fib_500'].values, name="fib_500")
        self.setup_high = self.I(map_to_indicator, self.data.df['setup_high'].values, name="setup_high")
        self.setup_low = self.I(map_to_indicator, self.data.df['setup_low'].values, name="setup_low")

    def next(self):
        if self.position or len(self.data.Close) < self.sl_lookback:
            return

        is_short_setup = self.setup_type[-1] == -1
        is_long_setup = self.setup_type[-1] == 1

        # --- Short Entry Logic ---
        if is_short_setup and self.data.High[-1] > self.fib_500[-1] and self.data.Close[-1] < self.fib_500[-1]:
            # Bearish engulfing confirmation
            if (self.data.Close[-1] < self.data.Open[-1] and
                self.data.Close[-2] > self.data.Open[-2] and
                self.data.Open[-1] > self.data.Close[-2] and
                self.data.Close[-1] < self.data.Open[-2]):

                recent_high = np.max(self.data.High[-self.sl_lookback:])
                sl_buffer = (recent_high - self.data.Close[-1]) * self.sl_buffer_multiplier
                sl = recent_high + sl_buffer

                retracement_range = recent_high - self.setup_low[-1]
                tp = recent_high - (retracement_range * 0.5)

                if tp < self.data.Close[-1]:
                    self.sell(sl=sl, tp=tp)

        # --- Long Entry Logic ---
        elif is_long_setup and self.data.Low[-1] < self.fib_500[-1] and self.data.Close[-1] > self.fib_500[-1]:
            # Bullish engulfing confirmation
            if (self.data.Close[-1] > self.data.Open[-1] and
                self.data.Close[-2] < self.data.Open[-2] and
                self.data.Open[-1] < self.data.Close[-2] and
                self.data.Close[-1] > self.data.Open[-2]):

                recent_low = np.min(self.data.Low[-self.sl_lookback:])
                sl_buffer = (self.data.Close[-1] - recent_low) * self.sl_buffer_multiplier
                sl = recent_low - sl_buffer

                retracement_range = self.setup_high[-1] - recent_low
                tp = recent_low + (retracement_range * 0.5)

                if tp > self.data.Close[-1]:
                    self.buy(sl=sl, tp=tp)

if __name__ == '__main__':
    data_1m = generate_synthetic_data(periods=10000)
    processed_data = preprocess_data(data_1m.copy(), prominence=1.5)

    bt = Backtest(processed_data, FibonacciMwInternalScalpStrategy, cash=10000, commission=.002)

    stats = bt.optimize(
        sl_buffer_multiplier=np.arange(0.5, 3.0, 0.5).tolist(),
        sl_lookback=range(3, 10, 2),
        maximize='Sharpe Ratio'
    )

    print("Best optimization results:")
    print(stats)

    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'fibonacci_mw_internal_scalp',
            'return': float(stats.get('Return [%]', 0.0)),
            'sharpe': float(stats.get('Sharpe Ratio', 0.0)),
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
            'win_rate': float(stats.get('Win Rate [%]', 0.0)),
            'total_trades': int(stats.get('# Trades', 0))
        }, f, indent=2)

    print("\nBacktest complete. Results saved to results/temp_result.json")

    bt.plot(filename="results/fibonacci_mw_internal_scalp_plot.html", open_browser=False)
    print("Plot saved to results/fibonacci_mw_internal_scalp_plot.html")
