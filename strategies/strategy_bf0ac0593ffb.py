
import pandas as pd
import numpy as np
import talib
from scipy.signal import find_peaks
from backtesting import Strategy, Backtest

class FibonacciRetracementExtensionStrategy(Strategy):
    # Optimizable parameters
    swing_lookback = 20  # Lookback period for identifying swing points
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_multiplier = 1.0

    def init(self):
        # Add indicators from preprocess_data to the strategy's context
        self.htf_trend_up = self.I(lambda: self.data.df['htf_trend_up'], name='htf_trend_up')
        self.volume_ma = self.I(lambda: self.data.df['volume_ma'], name='volume_ma')
        self.atr = self.I(lambda: self.data.df['atr'], name='atr')

        # Find swing points (peaks and troughs)
        highs = self.data.High
        lows = self.data.Low

        # Scipy's find_peaks identifies local maxima
        peak_indices, _ = find_peaks(highs, distance=self.swing_lookback)
        trough_indices, _ = find_peaks(-lows, distance=self.swing_lookback)

        # Store swing points for access in next()
        self.swing_highs = {i: highs[i] for i in peak_indices}
        self.swing_lows = {i: lows[i] for i in trough_indices}

    def next(self):
        current_index = len(self.data) - 1
        current_price = self.data.Close[-1]

        # --- FILTERS from GUIDELINES ---
        # 1. Higher Timeframe Trend Filter
        is_uptrend = self.htf_trend_up[-1]
        is_downtrend = not is_uptrend

        # 2. Volume Filter
        volume_ok = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_ma_multiplier
        if not volume_ok:
            return

        # --- ENTRY LOGIC ---
        if self.position:
            return

        # Find the most recent swing points
        relevant_high_idx = max([i for i in self.swing_highs.keys() if i < current_index], default=None)
        relevant_low_idx = max([i for i in self.swing_lows.keys() if i < current_index], default=None)

        if relevant_high_idx is None or relevant_low_idx is None:
            return

        swing_high_price = self.swing_highs[relevant_high_idx]
        swing_low_price = self.swing_lows[relevant_low_idx]

        # Long Entry Logic
        if is_uptrend and relevant_low_idx < relevant_high_idx:
            fib_range = swing_high_price - swing_low_price
            fib_382 = swing_high_price - 0.382 * fib_range
            fib_618 = swing_high_price - 0.618 * fib_range

            if current_price <= fib_382 and current_price >= fib_618:
                sl = current_price - self.atr[-1] * self.atr_sl_multiplier
                tp = current_price + self.atr[-1] * self.atr_tp_multiplier
                self.buy(sl=sl, tp=tp)

        # Short Entry Logic
        elif is_downtrend and relevant_high_idx < relevant_low_idx:
            fib_range = swing_low_price - swing_high_price
            fib_382 = swing_high_price + 0.382 * fib_range
            fib_618 = swing_high_price + 0.618 * fib_range

            if current_price >= fib_382 and current_price <= fib_618:
                sl = current_price + self.atr[-1] * self.atr_sl_multiplier
                tp = current_price - self.atr[-1] * self.atr_tp_multiplier
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    # --- DATA LOADING ---
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure the data file is in the correct directory.")
        exit()

    # --- PREPROCESSING ---
    data = preprocess_data(df)

    # --- SANITIZE DATA ---
    # Drop rows with NaN values that might have been created during preprocessing
    data = data.dropna()
    # Ensure OHLCV columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Input data must contain all required columns: {required_columns}")


    # --- BACKTESTING ---
    bt = Backtest(data, FibonacciRetracementExtensionStrategy, cash=100_000, commission=.002)
    stats = bt.run()
    print(stats)

    # --- OUTPUT ---
    # Save stats to a JSON file
    stats_dict = dict(stats)
    # The _strategy object is not serializable, so we remove it.
    if '_strategy' in stats_dict:
        del stats_dict['_strategy']

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)

    # Save plot to an HTML file
    bt.plot(filename='results/strategy_bf0ac0593ffb.html', open_browser=False)

    print("\nBacktest complete.")
    print(f"Stats saved to results/temp_result.json")
    print(f"Plot saved to results/strategy_bf0ac0593ffb.html")
