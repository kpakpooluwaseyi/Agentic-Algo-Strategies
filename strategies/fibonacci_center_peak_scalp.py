import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from backtesting import Backtest, Strategy

def preprocess_data(df_1m, prominence=1.5, peak_distance=5):
    """
    Pre-processes 1M data to identify Fibonacci setup signals from a 15M timeframe.
    """
    df_15m = df_1m['Close'].resample('15min').ohlc().dropna()

    # 1. Identify "Level Drop" - find significant peaks and troughs on 15M
    highs, _ = find_peaks(df_15m['high'], prominence=prominence, distance=peak_distance)
    lows, _ = find_peaks(-df_15m['low'], prominence=prominence, distance=peak_distance)

    signals = []
    for high_idx in highs:
        # Find the subsequent low that forms the "Level Drop"
        subsequent_lows = lows[lows > high_idx]
        if not subsequent_lows.any():
            continue
        low_idx = subsequent_lows[0]

        start_high_time = df_15m.index[high_idx]
        end_low_time = df_15m.index[low_idx]

        level_drop_high = df_15m['high'].iloc[high_idx]
        level_drop_low = df_15m['low'].iloc[low_idx]

        # 2. Calculate Fib levels for the initial drop
        fib_38 = level_drop_low + (level_drop_high - level_drop_low) * 0.382
        fib_50 = level_drop_low + (level_drop_high - level_drop_low) * 0.5
        fib_61 = level_drop_low + (level_drop_high - level_drop_low) * 0.618
        fib_78 = level_drop_low + (level_drop_high - level_drop_low) * 0.786

        signals.append({
            'start_time': start_high_time,
            'end_time': end_low_time,
            'level_drop_high': level_drop_high,
            'level_drop_low': level_drop_low,
            'fib_38': fib_38,
            'fib_50': fib_50,
            'fib_61': fib_61,
            'fib_78': fib_78,
        })

    if not signals:
        return df_1m

    signals_df = pd.DataFrame(signals).sort_values('start_time')

    # Use merge_asof for efficient mapping
    df_1m = pd.merge_asof(
        df_1m.sort_index(),
        signals_df.add_prefix('signal_'),
        left_index=True,
        right_on='signal_start_time',
        direction='backward'
    )
    df_1m = df_1m.drop(columns=['signal_start_time', 'signal_end_time']).ffill()

    return df_1m


class FibonacciCenterPeakScalpStrategy(Strategy):
    # --- Strategy Parameters ---
    min_rr = 5
    leverage = 1.0 # No leverage

    # --- State Variables ---
    setup_valid = False
    center_peak_high = 0
    invalidation_point_38 = 0
    invalidation_point_61 = 0
    invalidation_point_78 = 0
    aoi_50_level = 0
    initial_level_drop_low = 0

    def init(self):
        # Make the pre-processed data available to the strategy
        self.signal_level_drop_high = self.I(lambda x: x, self.data.df['signal_level_drop_high'])
        self.signal_level_drop_low = self.I(lambda x: x, self.data.df['signal_level_drop_low'])
        self.signal_fib_38 = self.I(lambda x: x, self.data.df['signal_fib_38'])
        self.signal_fib_50 = self.I(lambda x: x, self.data.df['signal_fib_50'])
        self.signal_fib_61 = self.I(lambda x: x, self.data.df['signal_fib_61'])
        self.signal_fib_78 = self.I(lambda x: x, self.data.df['signal_fib_78'])
        self.setup_valid = False

    def next(self):
        # If a position is already open, do nothing.
        if self.position:
            return

        # --- State Management & Setup Validation ---

        # A change in the signal high indicates a new 15M setup is available
        if self.signal_level_drop_high[-1] != self.signal_level_drop_high[-2]:
            self.setup_valid = True
            self.center_peak_high = 0 # Reset center peak on new setup
            self.aoi_50_level = self.signal_fib_50[-1]
            self.initial_level_drop_low = self.signal_level_drop_low[-1]
            self.invalidation_point_38 = self.signal_fib_38[-1]
            self.invalidation_point_61 = self.signal_fib_61[-1]
            self.invalidation_point_78 = self.signal_fib_78[-1]

        if not self.setup_valid:
            return

        # Invalidation: If price touches other Fib levels before the 50%
        if self.data.High[-1] >= self.invalidation_point_61 or \
           self.data.High[-1] >= self.invalidation_point_78 or \
           self.data.Low[-1] <= self.invalidation_point_38:
            self.setup_valid = False
            return

        # --- Entry Logic ---

        # Price must touch the 50% AOI
        price_reached_aoi = self.data.High[-1] >= self.aoi_50_level and self.data.Low[-1] <= self.aoi_50_level

        if price_reached_aoi:
            # Update the highest point of the retracement (Center Peak)
            self.center_peak_high = max(self.center_peak_high, self.data.High[-1])

            # Confirmation: Bearish Engulfing Candle
            is_bearish_engulfing = (self.data.Close[-1] < self.data.Open[-1] and
                                     self.data.Open[-2] < self.data.Close[-2] and # Previous candle is bullish
                                     self.data.Open[-1] > self.data.Close[-2] and
                                     self.data.Close[-1] < self.data.Open[-2])

            if is_bearish_engulfing:
                # --- Risk Management ---
                stop_loss = self.center_peak_high * 1.0005 # SL slightly above the peak

                # Exit rule: TP is 50% of the *second* Fib measurement
                take_profit = self.initial_level_drop_low + (self.center_peak_high - self.initial_level_drop_low) * 0.5

                # Enforce R:R
                risk = abs(self.data.Close[-1] - stop_loss)
                reward = abs(self.data.Close[-1] - take_profit)
                if risk == 0 or reward / risk < self.min_rr:
                    return

                # Position Sizing
                size = (self.equity * self.leverage * 0.01) / risk # Risk 1% of equity

                # --- Place Trade ---
                self.sell(sl=stop_loss, tp=take_profit, size=size)
                self.setup_valid = False # Invalidate setup after entry

def generate_synthetic_data(periods=2000):
    """
    Generates synthetic 1M data that intentionally creates the
    Fibonacci Center Peak Scalp pattern.
    """
    rng = np.random.default_rng(42)
    ts = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.arange(periods), 'min')
    price = np.ones(periods) * 100

    # --- Pattern 1: A clear drop, retracement, and continuation ---

    # 1. Level Drop
    high_1 = 105
    low_1 = 100
    price[100:130] = np.linspace(high_1, low_1, 30)

    # 2. Consolidation after drop
    price[130:160] = low_1 + rng.uniform(-0.1, 0.1, 30).cumsum()

    # 3. Retracement to the 50% Fib level (Center Peak)
    center_peak_high = low_1 + (high_1 - low_1) * 0.5  # 102.5
    price[160:190] = np.linspace(price[159], center_peak_high, 30)

    # 4. Bearish reversal candle at the peak
    price[190] = center_peak_high - 0.1 # Open
    price[191] = center_peak_high + 0.05 # High
    price[192] = center_peak_high - 0.8 # Low and Close, making a bearish candle
    price[193:250] = price[192] + rng.uniform(-0.1, 0.1, 57).cumsum()

    # 5. Second Drop towards TP
    tp_level = low_1 + (center_peak_high - low_1) * 0.5 # 101.25
    price[250:280] = np.linspace(price[249], tp_level, 30)

    # Fill remaining with some noise
    price[280:] = price[279] + rng.uniform(-0.1, 0.1, periods - 280).cumsum()

    # Create DataFrame
    df = pd.DataFrame({'Open': price, 'High': price, 'Low': price, 'Close': price}, index=ts)

    # Add some candle-like variations
    df['Open'] = df['Close'].shift(1).fillna(method='bfill')
    df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0, 0.1, periods)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0, 0.1, periods)

    # Ensure reversal candle is bearish engulfing
    df.loc[df.index[191], 'Open'] = center_peak_high - 0.1
    df.loc[df.index[191], 'High'] = center_peak_high + 0.05
    df.loc[df.index[191], 'Close'] = center_peak_high - 0.8
    df.loc[df.index[191], 'Low'] = center_peak_high - 0.81

    # Previous candle
    df.loc[df.index[190], 'Open'] = center_peak_high - 0.3
    df.loc[df.index[190], 'Close'] = center_peak_high - 0.2

    return df.dropna()


if __name__ == '__main__':
    data = generate_synthetic_data()

    # Pre-process data
    # Note: Optimization of pre-processing params requires a custom loop.
    # For this example, we'll use fixed pre-processing params.
    preprocessed_data = preprocess_data(data, prominence=1.5, peak_distance=5)

    # Run backtest
    bt = Backtest(preprocessed_data, FibonacciCenterPeakScalpStrategy, cash=100_000, commission=.002)

    # Optimize
    stats = bt.optimize(
        min_rr=range(3, 8, 1),
        leverage=[1.0], # Keep leverage fixed for this test
        maximize='Sharpe Ratio'
    )

    print(stats)

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Clean up potential NaN values from stats
    sharpe = stats['Sharpe Ratio'] if not np.isnan(stats['Sharpe Ratio']) else 0.0

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'fibonacci_center_peak_scalp',
            'return': float(stats['Return [%]']),
            'sharpe': float(sharpe),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    # Generate plot
    bt.plot(filename='results/fibonacci_center_peak_scalp.html', open_browser=False)
