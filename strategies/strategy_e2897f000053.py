from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

def generate_synthetic_data(days=100):
    """
    Generates synthetic 15-minute data with clear range-overextension-reversal patterns.
    This helps verify the strategy's core logic.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range(start='2023-01-01', periods=days * 24 * 4, freq='15min')
    df = pd.DataFrame(index=dates)

    price = 100
    prices = [price]
    for _ in range(len(dates) - 1):
        price += rng.normal(0, 0.5)
        prices.append(price)
    df['Close'] = pd.Series(prices, index=dates).rolling(window=5, min_periods=1).mean()

    # Inject specific overextension patterns
    for day in range(5, days, 15):
        # Define a clear range
        range_start_time = pd.to_datetime(f'2023-01-01') + pd.Timedelta(days=day, hours=0)
        range_end_time = range_start_time + pd.Timedelta(hours=10)
        range_mask = (df.index >= range_start_time) & (df.index <= range_end_time)
        range_base = df.loc[range_start_time, 'Close']
        df.loc[range_mask, 'Close'] = range_base + np.sin(np.linspace(0, 2 * np.pi, range_mask.sum())) * 5

        # Create overextension and reversal
        ext_start_time = range_end_time + pd.Timedelta(minutes=15)
        ext_peak_time = ext_start_time + pd.Timedelta(minutes=15)
        rev_time = ext_peak_time + pd.Timedelta(minutes=15)

        range_high = df.loc[range_mask, 'Close'].max()
        df.loc[ext_start_time, 'Close'] = range_high + 1
        df.loc[ext_peak_time, 'Close'] = range_high + 3 # Overextension peak
        df.loc[rev_time, 'Close'] = range_high - 2     # Sharp reversal

    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0.1, 0.5, size=len(df))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0.1, 0.5, size=len(df))
    df['Volume'] = rng.integers(100, 1000, size=len(df))
    df = df.dropna()
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def preprocess_data(df, range_hours=8):
    """
    Calculates the rolling trading range and adds time-based features for simulation.
    """
    rolling_window_size = range_hours * 4 # 8 hours * 4 quarters/hour

    # Define the context range using a rolling window on past data
    df['range_high'] = df['High'].shift(1).rolling(window=rolling_window_size, min_periods=rolling_window_size).max()
    df['range_low'] = df['Low'].shift(1).rolling(window=rolling_window_size, min_periods=rolling_window_size).min()

    # Add minute_of_hour to simulate intra-hour candle behavior
    df['minute_of_hour'] = df.index.minute

    df = df.dropna()
    return df

def passthrough(data):
    return data

from enum import Enum

class State(Enum):
    SEARCHING = 0
    OVEREXTENDED_HIGH = 1
    OVEREXTENDED_LOW = 2
    SHIFTED_HIGH = 3
    SHIFTED_LOW = 4

class CandleBehaviorReversalStrategy(Strategy):
    range_hours = 10
    overextension_min_minutes = 20
    overextension_max_minutes = 45
    timeout_bars = 4 # Number of 15m bars to wait for a shift (1 hour)

    def init(self):
        self.range_high = self.I(passthrough, self.data.df['range_high'])
        self.range_low = self.I(passthrough, self.data.df['range_low'])
        self.minute_of_hour = self.I(passthrough, self.data.df['minute_of_hour'])
        self.reset_state()

    def next(self):
        if self.position:
            return
        if self.state == State.SEARCHING:
            self.search_for_overextension()
        elif self.state == State.OVEREXTENDED_HIGH:
            self.monitor_for_shift_after_high()
        elif self.state == State.OVEREXTENDED_LOW:
            self.monitor_for_shift_after_low()

    def search_for_overextension(self):
        is_new_hour = self.minute_of_hour[-1] == 0
        if is_new_hour and self.range_low[-1] < self.data.Open[-1] < self.range_high[-1]:
            self.hourly_context_idx = len(self.data.Close) - 1

        if self.hourly_context_idx is not None:
            if is_new_hour and (len(self.data.Close) - 1 > self.hourly_context_idx):
                self.reset_state()
                return

            time_diff = (self.data.index[-1] - self.data.index[self.hourly_context_idx]).total_seconds() / 60
            is_in_time_window = self.overextension_min_minutes <= time_diff <= self.overextension_max_minutes

            if is_in_time_window:
                if self.data.High[-1] > self.range_high[-1]:
                    self.state = State.OVEREXTENDED_HIGH
                    self.overextension_peak = self.data.High[-1]
                    self.shift_low_point = self.data.Low[-1]
                    self.timeout_idx = len(self.data.Close) - 1
                elif self.data.Low[-1] < self.range_low[-1]:
                    self.state = State.OVEREXTENDED_LOW
                    self.overextension_peak = self.data.Low[-1]
                    self.shift_high_point = self.data.High[-1]
                    self.timeout_idx = len(self.data.Close) - 1

    def monitor_for_shift_after_high(self):
        if len(self.data.Close) -1 > self.timeout_idx + self.timeout_bars:
            self.reset_state()
            return

        if self.data.High[-1] > self.overextension_peak:
            self.overextension_peak = self.data.High[-1]
            self.shift_low_point = self.data.Low[-1]
            self.timeout_idx = len(self.data.Close) -1 # Reset timeout
            return

        if self.shift_low_point and self.data.Close[-1] < self.shift_low_point:
            entry_price = self.data.Close[-1]
            sl = self.overextension_peak
            tp = self.overextension_peak - (self.overextension_peak - self.range_high[-1]) * 0.5
            if sl > entry_price and tp < entry_price:
                self.sell(sl=sl, tp=tp)
            self.reset_state()

    def monitor_for_shift_after_low(self):
        if len(self.data.Close) -1 > self.timeout_idx + self.timeout_bars:
            self.reset_state()
            return

        if self.data.Low[-1] < self.overextension_peak:
            self.overextension_peak = self.data.Low[-1]
            self.shift_high_point = self.data.High[-1]
            self.timeout_idx = len(self.data.Close) -1 # Reset timeout
            return

        if self.shift_high_point and self.data.Close[-1] > self.shift_high_point:
            entry_price = self.data.Close[-1]
            sl = self.overextension_peak
            tp = self.overextension_peak + (self.range_low[-1] - self.overextension_peak) * 0.5
            if sl < entry_price and tp > entry_price:
                self.buy(sl=sl, tp=tp)
            self.reset_state()

    def reset_state(self):
        self.state = State.SEARCHING
        self.hourly_context_idx = None
        self.overextension_peak = None
        self.shift_low_point = None
        self.shift_high_point = None
        self.timeout_idx = None

if __name__ == '__main__':
    import os

    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        column_names = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = pd.read_csv(
            data_path, header=0, names=column_names,
            index_col='datetime', parse_dates=True, usecols=column_names
        )
    else:
        print("Historical data not found. Generating synthetic data...")
        data = generate_synthetic_data(days=200)

    data = preprocess_data(data)

    from backtesting.lib import FractionalBacktest
    bt = FractionalBacktest(data, CandleBehaviorReversalStrategy, cash=100000, commission=.002, finalize_trades=True)

    print("Running backtest with tuned parameters...")
    stats = bt.run(range_hours=4, overextension_min_minutes=5, overextension_max_minutes=60)

    # Save results
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        # A helper function to remove non-serializable types from the stats dictionary
        sanitized = {}
        for key, value in stats.items():
            # Skip internal objects which are often not serializable (like _strategy)
            if key.startswith('_'):
                continue
            if isinstance(value, (pd.Series, pd.DataFrame, Strategy, type(pd.NA))):
                continue
            elif pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (np.integer, np.int64)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sanitized[key] = float(value)
            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                 sanitized[key] = str(value)
            else:
                sanitized[key] = value
        return sanitized

    clean_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(clean_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    try:
        bt.plot(filename='results/strategy_e2897f000053_plot.html')
        print("Plot saved to results/strategy_e2897f000053_plot.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
