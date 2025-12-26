import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks
import json
import os

def is_pin_bar(candle):
    """ Detects a Pin Bar candlestick pattern. """
    body_size = abs(candle.Close - candle.Open)
    if body_size == 0: return False, False
    lower_wick = min(candle.Open, candle.Close) - candle.Low
    upper_wick = candle.High - max(candle.Open, candle.Close)
    is_bullish = lower_wick > 2 * body_size and upper_wick < body_size
    is_bearish = upper_wick > 2 * body_size and lower_wick < body_size
    return is_bullish, is_bearish

def is_engulfing(candle, prev_candle):
    """ Detects an Engulfing candlestick pattern. """
    is_bullish = candle.Close > prev_candle.Open and candle.Open < prev_candle.Close and \
                 candle.Close > candle.Open and prev_candle.Close < prev_candle.Open
    is_bearish = candle.Close < prev_candle.Open and candle.Open > prev_candle.Close and \
                 candle.Close < candle.Open and prev_candle.Close > prev_candle.Open
    return is_bullish, is_bearish

def find_sr_zones(series_np, distance, is_resistance):
    """Finds support/resistance zones and returns a series of NaNs with values at peak/trough indices."""
    series_pd = pd.Series(series_np)
    if not is_resistance:
        series_for_peaks = -series_pd
    else:
        series_for_peaks = series_pd

    peak_indices, _ = find_peaks(series_for_peaks, distance=distance)

    output_array = np.full(len(series_np), np.nan)
    output_array[peak_indices] = series_np[peak_indices]
    return output_array

class sr_macd_stochrsi_reversal(Strategy):
    # Strategy parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    stoch_k = 14
    stoch_d = 3
    stoch_rsi_len = 14
    sr_distance = 20 # Increased to find more significant S/R levels
    sl_buffer_pct = 0.02 # Increased buffer
    macd_gap_threshold = 1 # Adjusted for BTC price scale
    sr_proximity_pct = 0.02 # Added proximity parameter

    def init(self):
        close_series = pd.Series(self.data.Close)

        # MACD
        macd_df = ta.macd(close=close_series, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal, append=True)
        self.macd = self.I(lambda: macd_df.iloc[:, 0])
        self.macd_signal = self.I(lambda: macd_df.iloc[:, 1])
        self.macd_hist = self.I(lambda: macd_df.iloc[:, 2])

        # Stochastic RSI
        stochrsi_df = ta.stochrsi(close=close_series, length=self.stoch_rsi_len, rsi_length=self.stoch_rsi_len, k=self.stoch_k, d=self.stoch_d, append=True)
        self.stoch_rsi = self.I(lambda: stochrsi_df.iloc[:, 0])
        self.stoch_rsi_signal = self.I(lambda: stochrsi_df.iloc[:, 1])

        # Volume
        self.volume = self.I(lambda: self.data.Volume)

        # S/R zones
        self.resistance = self.I(find_sr_zones, self.data.High, self.sr_distance, True)
        self.support = self.I(find_sr_zones, self.data.Low, self.sr_distance, False)

        # State machine for delayed entry
        self.signal_bar_index = -1
        self.trade_direction = None
        self.sl_price = None
        self.tp_price = None

    def next(self):
        current_bar_index = len(self.data) - 1

        # Delayed entry logic
        if self.signal_bar_index != -1 and current_bar_index == self.signal_bar_index + 2:
            if self.trade_direction == 'long' and self.tp_price > self.data.Close[-1]:
                self.buy(sl=self.sl_price, tp=self.tp_price)
            elif self.trade_direction == 'short' and self.tp_price < self.data.Close[-1]:
                self.sell(sl=self.sl_price, tp=self.tp_price)

            # Reset state
            self.signal_bar_index = -1
            self.trade_direction = None

        if self.position or self.signal_bar_index != -1:
            return

        # Entry signal detection
        current_candle = self.data.df.iloc[-1]
        prev_candle = self.data.df.iloc[-2]

        # Reversal patterns
        pin_bull, pin_bear = is_pin_bar(current_candle)
        eng_bull, eng_bear = is_engulfing(current_candle, prev_candle)
        is_reversal_bull = pin_bull or eng_bull
        is_reversal_bear = pin_bear or eng_bear

        # S/R proximity
        last_support = np.nanmax(self.support.to_series().fillna(-np.inf).values)
        last_resistance = np.nanmin(self.resistance.to_series().fillna(np.inf).values)

        at_support = abs(self.data.Low[-1] - last_support) / last_support < self.sr_proximity_pct
        at_resistance = abs(self.data.High[-1] - last_resistance) / last_resistance < self.sr_proximity_pct

        # Volume
        volume_declining = self.volume[-1] < self.volume[-2]

        # MACD
        macd_gap = abs(self.macd_hist[-1]) > self.macd_gap_threshold
        macd_about_to_cross_bull = self.macd[-1] < self.macd_signal[-1] and self.macd[-1] > self.macd[-2]
        macd_about_to_cross_bear = self.macd[-1] > self.macd_signal[-1] and self.macd[-1] < self.macd[-2]

        # Stoch RSI
        stoch_oversold_cross = self.stoch_rsi[-1] < 20 and crossover(self.stoch_rsi, self.stoch_rsi_signal)
        stoch_overbought_cross = self.stoch_rsi[-1] > 80 and crossover(self.stoch_rsi_signal, self.stoch_rsi)

        # Long entry conditions
        if is_reversal_bull and at_support and stoch_oversold_cross:
            self.signal_bar_index = current_bar_index
            self.trade_direction = 'long'
            self.sl_price = self.data.Low[-1] * (1 - self.sl_buffer_pct)
            self.tp_price = last_resistance

        # Short entry conditions
        elif is_reversal_bear and at_resistance and stoch_overbought_cross:
            self.signal_bar_index = current_bar_index
            self.trade_direction = 'short'
            self.sl_price = self.data.High[-1] * (1 + self.sl_buffer_pct)
            self.tp_price = last_support

if __name__ == '__main__':
    data = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
    data.columns = [c.strip().title() for c in data.columns]

    bt = Backtest(data, sr_macd_stochrsi_reversal, cash=100_000, commission=.002)

    stats = bt.run()
    print(stats)

    # Save results
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats_obj):
        if isinstance(stats_obj, pd.Series):
            stats_dict = stats_obj.to_dict()
        else:
            stats_dict = dict(stats_obj)

        stats_dict.pop('_strategy', None)
        stats_dict.pop('_equity_curve', None)
        stats_dict.pop('_trades', None)

        sanitized_dict = {}
        for key, value in stats_dict.items():
            if pd.isna(value):
                sanitized_dict[key] = None
            elif isinstance(value, (np.integer, int)):
                sanitized_dict[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                sanitized_dict[key] = float(value)
            elif isinstance(value, pd.Timestamp):
                sanitized_dict[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                sanitized_dict[key] = str(value)
            else:
                sanitized_dict[key] = value
        return sanitized_dict

    sanitized_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=2)

    bt.plot(filename='results/sr_macd_stochrsi_reversal.html')
