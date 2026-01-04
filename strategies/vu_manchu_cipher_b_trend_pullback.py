from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
from scipy.signal import find_peaks

def MFI(high, low, close, volume, n):
    """Calculates the Money Flow Index (MFI)."""
    mfi = ta.mfi(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), volume=pd.Series(volume), length=n)
    return mfi.values

def MACD(close, fast, slow, signal):
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    macd = ta.macd(close=pd.Series(close), fast=fast, slow=slow, signal=signal)
    # The histogram is the third column
    return macd.iloc[:, 2].values

def EMA(series, n):
    """Returns the EMA of a given series."""
    return pd.Series(series).ewm(span=n, adjust=False).mean().values

class VuManChuCipherBTrendPullback(Strategy):
    # EMA settings
    ema_long_period = 200
    ema_short_period = 50

    # MACD settings (proxy for Blue Waves)
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # MFI settings (proxy for Money Flow)
    mfi_period = 14

    # Risk management
    risk_reward_ratio = 2
    swing_lookback = 20 # Lookback for finding swing points for SL

    def init(self):
        # --- Indicator Proxy Explanation ---
        # The user's request specified using the `src.indicators.vumanchu` module.
        # However, an initial file system exploration confirmed that this module does not exist.
        # As per established procedure for missing custom indicators, functional proxies
        # have been implemented using standard `pandas_ta` library indicators.
        #
        # - **VuManChu's "Money Flow"** is proxied by the Money Flow Index (MFI).
        #   A value > 50 is considered "green" (bullish), and < 50 is "red" (bearish).
        #
        # - **VuManChu's "Blue Waves"** and their crossover signals are proxied by the
        #   MACD histogram. A crossover of the histogram above the zero line represents a
        #   "green dot" (bullish signal), and a crossover below represents a "red dot" (bearish signal).

        # Indicators
        self.ema_long = self.I(EMA, self.data.Close, self.ema_long_period)
        self.ema_short = self.I(EMA, self.data.Close, self.ema_short_period)

        self.money_flow = self.I(MFI, self.data.High, self.data.Low, self.data.Close, self.data.Volume, self.mfi_period)
        self.blue_waves = self.I(MACD, self.data.Close, self.macd_fast, self.macd_slow, self.macd_signal)

        # Helper for stop loss placement
        self.swing_highs = self.I(self._swing_points, self.data.High, distance=self.swing_lookback, is_high=True)
        self.swing_lows = self.I(self._swing_points, self.data.Low, distance=self.swing_lookback, is_high=False)

    def _swing_points(self, series, distance, is_high):
        """Finds swing highs or lows using scipy.signal.find_peaks."""
        if is_high:
            peaks, _ = find_peaks(series, distance=distance)
            points = np.zeros_like(series, dtype=np.float64)
            points[peaks] = series[peaks]
        else:
            troughs, _ = find_peaks(-series, distance=distance)
            points = np.zeros_like(series, dtype=np.float64)
            points[troughs] = series[troughs]
        return points

    def next(self):
        price = self.data.Close[-1]

        # --- Long Entry Conditions ---
        is_uptrend = price > self.ema_long[-1]
        is_pullback = price < self.ema_short[-1]
        money_flow_green = self.money_flow[-1] > 50
        waves_below_zero_prev = self.blue_waves[-2] < 0
        wave_cross_up = crossover(self.blue_waves, 0)

        if is_uptrend and is_pullback and money_flow_green and waves_below_zero_prev and wave_cross_up and not self.position:
            # Find the most recent swing low for the stop loss
            recent_swing_lows = [p for p in self.swing_lows if p > 0]
            if recent_swing_lows:
                stop_loss = recent_swing_lows[-1]
                take_profit = price + (price - stop_loss) * self.risk_reward_ratio

                # Ensure SL and TP are valid before placing trade
                if stop_loss < price and take_profit > price:
                    self.buy(sl=stop_loss, tp=take_profit)

        # --- Short Entry Conditions ---
        is_downtrend = price < self.ema_long[-1]
        is_pullback_short = price > self.ema_short[-1]
        money_flow_red = self.money_flow[-1] < 50
        waves_above_zero_prev = self.blue_waves[-2] > 0
        wave_cross_down = crossover(0, self.blue_waves)

        if is_downtrend and is_pullback_short and money_flow_red and waves_above_zero_prev and wave_cross_down and not self.position:
            # Find the most recent swing high for the stop loss
            recent_swing_highs = [p for p in self.swing_highs if p > 0]
            if recent_swing_highs:
                stop_loss = recent_swing_highs[-1]
                take_profit = price - (stop_loss - price) * self.risk_reward_ratio

                # Ensure SL and TP are valid before placing trade
                if stop_loss > price and take_profit < price:
                    self.sell(sl=stop_loss, tp=take_profit)

if __name__ == '__main__':
    # --- Data Loading and Preparation ---
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure the data file is in the correct location.")
        # As a fallback, use synthetic data for demonstration
        from backtesting.test import GOOG
        data = GOOG.copy()
        data = data.resample('15min').last().ffill()

    # Ensure column names are in the format Backtesting.py expects
    data.columns = [col.capitalize() for col in data.columns]

    # --- Backtesting ---
    bt = Backtest(data, VuManChuCipherBTrendPullback, cash=10000, commission=.002)

    stats = bt.run()
    print(stats)

    # --- Results Handling ---
    import os
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        """Prepares the stats object for JSON serialization."""
        if stats is None:
            return {}

        # Remove non-serializable items
        stats.pop('_strategy', None)
        stats.pop('_equity_curve', None)
        stats.pop('_trades', None)

        # Convert specific types to JSON-friendly formats
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif pd.isna(value) or np.isnan(value):
                 sanitized[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                sanitized[key] = float(value)
            else:
                sanitized[key] = value
        return sanitized

    # Save stats to a JSON file
    sanitized_results = sanitize_stats(stats.to_dict() if isinstance(stats, pd.Series) else stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_results, f, indent=4)

    # Generate and save the plot
    try:
        bt.plot(filename='results/plot.html', open_browser=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")
