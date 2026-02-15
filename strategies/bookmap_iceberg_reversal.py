
import json
import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.signal import find_peaks

from backtesting import Backtest, Strategy
from src.indicators.vumanchu import cipher_b


def preprocess_data(df, timeframe='4h', ema_length=50, **params):
    """
    Applies indicator calculations to the input DataFrame.
    - Calculates a higher-timeframe EMA for trend filtering.
    """
    # Ensure we are working with a copy
    df = df.copy()

    # Calculate higher-timeframe EMA
    ema_htf = ta.ema(df['Close'].resample(timeframe).last(), length=ema_length)

    # Reindex the HTF EMA to the original dataframe's index and forward-fill
    df['ema_4h'] = ema_htf.reindex(df.index, method='ffill')

    # Drop rows where the ema_4h is NaN (due to EMA warm-up period)
    df = df.dropna(subset=['ema_4h'])

    return df


class BookmapIcebergReversal(Strategy):
    # --- Strategy Parameters ---
    wfa_hyperparameters = {
        "peak_prominence": [0.005, 0.01, 0.02],
        "mfi_threshold": [20, 30, 40],
        "atr_multiplier_tp": [2, 3, 4],
        "atr_multiplier_sl": [1.5, 2, 2.5],
    }

    # --- Strategy Parameters ---
    wfa_hyperparameters = {
        "peak_prominence": [0.01, 0.02, 0.03],
        "volume_multiplier": [1.5, 2.0, 2.5],
        "wick_to_body_ratio": [1.5, 2.0, 2.5],
        "body_max_percent": [0.01, 0.02, 0.03], # This is a key parameter to tune
    }

    peak_prominence = 0.02
    volume_multiplier = 2.0
    wick_to_body_ratio = 2.0
    body_max_percent = 0.03 # Relaxed from 0.01 to allow more signals
    atr_multiplier_tp = 3
    atr_multiplier_sl = 2

    def init(self):
        # --- Indicators ---
        self.atr = self.I(self.get_atr, self.data.High, self.data.Low, self.data.Close, length=14, name="ATR")

        # Pre-calculate swing lows to use as key support levels
        self.swing_lows = self.I(self.get_swing_lows, self.data.Close, prominence=self.peak_prominence, name="swing_lows")

        # Higher timeframe trend filter (pre-calculated in preprocess_data)
        self.ema_4h = self.I(lambda: self.data.df['ema_4h'], name="ema_4h")
        self.volume_ma = self.I(self.get_sma, self.data.Volume, length=20, name="volume_ma")

        # Proxy for Iceberg Order Detection: Hammer Candlestick
        self.is_hammer = self.I(self.get_hammer, self.data.Open, self.data.High, self.data.Low, self.data.Close,
                                self.wick_to_body_ratio, self.body_max_percent, name="is_hammer")

    def next(self):
        if len(self.trades) > 0:
            return

        # --- Iceberg Order Proxy Logic ---
        # The core idea is to identify a Hammer candlestick pattern occurring at a key support level (swing low)
        # on significantly higher-than-average volume. This combination proxies the visual absorption signature of
        # a hidden iceberg buy order as seen on a Bookmap heatmap. The key signal is a Hammer candle on high volume,
        # indicating strong absorption of selling pressure.

        # Condition 1: A Hammer candle forms, signaling rejection of lower prices.
        is_hammer_candle = self.is_hammer[-1] > 0

        # Condition 2: Volume is significantly higher than average, confirming strong buying pressure (absorption).
        is_high_volume = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_multiplier

        # Condition 3: The broader market trend is upwards (price is above 4H EMA).
        is_uptrend = self.data.Close[-1] > self.data.ema_4h[-1]

        if is_hammer_candle and is_high_volume and is_uptrend:
            # Execute long trade
            sl = self.data.Low[-1] - self.atr[-1] * 0.5 # Tighter stop below the hammer's low
            tp = self.data.Close[-1] + self.atr[-1] * self.atr_multiplier_tp
            self.buy(sl=sl, tp=tp)

    @staticmethod
    def get_swing_lows(data, prominence, **kwargs):
        peaks, _ = find_peaks(-np.array(data), prominence=prominence)
        signals = np.zeros(len(data))
        signals[peaks] = 1
        return signals

    @staticmethod
    def get_atr(high, low, close, length, **kwargs):
        """Wrapper for pandas_ta.atr to handle numpy array inputs."""
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        atr = ta.atr(high=high_series, low=low_series, close=close_series, length=length)
        return atr.values

    @staticmethod
    def get_sma(data, length, **kwargs):
        """Wrapper for pandas_ta.sma to handle numpy array inputs."""
        series = pd.Series(data)
        sma = ta.sma(series, length=length)
        return sma.values

    @staticmethod
    def get_hammer(open, high, low, close, wick_to_body_ratio, body_max_percent, **kwargs):
        """
        Identifies Hammer candles.
        A Hammer is a bullish reversal pattern with a small body near the top and a long lower wick.
        """
        body_size = np.abs(close - open)
        lower_wick = np.minimum(open, close) - low
        upper_wick = high - np.maximum(open, close)

        # A small body relative to the close price is a more stable measure than total range
        is_small_body = body_size <= close * body_max_percent
        is_long_lower_wick = lower_wick >= body_size * wick_to_body_ratio
        is_small_upper_wick = upper_wick < body_size

        return is_small_body & is_long_lower_wick & is_small_upper_wick


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    df = pd.read_csv(data_path)
    df.columns = [col.strip().capitalize() for col in df.columns]
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')

    # Preprocess data to add indicators
    df = preprocess_data(df)

    # --- Backtest ---
    bt = Backtest(df, BookmapIcebergReversal, cash=100000, commission=.002, finalize_trades=True)

    stats = bt.run()
    print(stats)

    # --- Save results ---
    if not os.path.exists('results'):
        os.makedirs('results')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_base = f"bookmap_iceberg_reversal_{timestamp}"

    bt.plot(filename=f"results/{filename_base}.html")

    # --- Prepare stats for JSON output ---
    results_dict = stats.to_dict()

    # Manually handle non-serializable or complex objects
    if '_strategy' in results_dict:
        results_dict['_strategy'] = str(results_dict['_strategy'])
    if '_equity_curve' in results_dict:
        equity_curve = results_dict['_equity_curve']
        equity_curve.index = equity_curve.index.astype(str)
        results_dict['_equity_curve'] = equity_curve.to_dict()
    if '_trades' in results_dict:
        results_dict['_trades'] = results_dict['_trades'].to_dict('records')

    def json_serial_helper(o):
        """Helper function to serialize types that json doesn't handle by default."""
        if isinstance(o, (pd.Timestamp, pd.Timedelta)):
            return str(o)
        if o is pd.NaT:
            return None
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(f"results/temp_result.json", 'w') as f:
        json.dump(results_dict, f, indent=4, default=json_serial_helper)

    print(f"Results saved to results/temp_result.json")
