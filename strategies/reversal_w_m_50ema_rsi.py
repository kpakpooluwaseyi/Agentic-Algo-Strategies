from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import json
import os

def RSI(series, n):
    """Calculate Relative Strength Index (RSI) using EMA"""
    series = pd.Series(series)
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def is_hammer(candle):
    """ Detects a Hammer candlestick pattern. """
    body_size = abs(candle.Close - candle.Open)
    if body_size == 0: return False
    lower_wick = min(candle.Open, candle.Close) - candle.Low
    upper_wick = candle.High - max(candle.Open, candle.Close)
    return lower_wick > 2 * body_size and upper_wick < body_size

def is_inverted_hammer(candle):
    """ Detects an Inverted Hammer candlestick pattern. """
    body_size = abs(candle.Close - candle.Open)
    if body_size == 0: return False
    upper_wick = candle.High - max(candle.Open, candle.Close)
    lower_wick = min(candle.Open, candle.Close) - candle.Low
    return upper_wick > 2 * body_size and lower_wick < body_size

def find_patterns(series_np, distance, is_troughs):
    """Finds peaks/troughs and returns a series of NaNs with values at peak/trough indices."""
    series_pd = pd.Series(series_np)
    if is_troughs:
        series_for_peaks = -series_pd
    else:
        series_for_peaks = series_pd

    peak_indices, _ = find_peaks(series_for_peaks, distance=distance)

    output_array = np.full(len(series_np), np.nan)
    output_array[peak_indices] = series_np[peak_indices]
    return output_array

class ReversalWM50EmaRsiStrategy(Strategy):
    rsi_period = 14
    ema_10_len = 10
    ema_20_len = 20
    ema_50_len = 50
    ema_200_len = 200
    ema_800_len = 800
    distance = 5

    def init(self):
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
        self.ema_10 = self.I(lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean(), self.data.Close, self.ema_10_len)
        self.ema_20 = self.I(lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean(), self.data.Close, self.ema_20_len)
        self.ema_50 = self.I(lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean(), self.data.Close, self.ema_50_len)
        self.ema_200 = self.I(lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean(), self.data.Close, self.ema_200_len)
        self.ema_800 = self.I(lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean(), self.data.Close, self.ema_800_len)
        self.troughs = self.I(find_patterns, self.data.Low, self.distance, True)
        self.peaks = self.I(find_patterns, self.data.High, self.distance, False)

    def next(self):
        price = self.data.Close[-1]

        trough_indices = np.where(~np.isnan(self.troughs))[0]
        is_w = False
        if len(trough_indices) >= 2:
            t1_idx, t2_idx = trough_indices[-2], trough_indices[-1]
            if len(self.data) - 1 - t2_idx < self.distance: # Check if pattern is recent
                t1_low, t2_low = self.data.Low[t1_idx], self.data.Low[t2_idx]
                if abs(t1_low - t2_low) / t1_low < 0.02 and is_hammer(self.data.df.iloc[t2_idx]):
                    is_w = True

        peak_indices = np.where(~np.isnan(self.peaks))[0]
        is_m = False
        if len(peak_indices) >= 2:
            p1_idx, p2_idx = peak_indices[-2], peak_indices[-1]
            if len(self.data) - 1 - p2_idx < self.distance: # Check if pattern is recent
                p1_high, p2_high = self.data.High[p1_idx], self.data.High[p2_idx]
                if abs(p1_high - p2_high) / p1_high < 0.02 and is_inverted_hammer(self.data.df.iloc[p2_idx]):
                    is_m = True

        retest_long = self.data.Low[-2] <= self.ema_50[-2] and self.data.Close[-2] > self.ema_50[-2]
        retest_short = self.data.High[-2] >= self.ema_50[-2] and self.data.Close[-2] < self.ema_50[-2]

        if not self.position:
            if is_w and crossover(self.rsi, 60) and price > self.ema_50[-1] and retest_long:
                sl, tp1, tp2 = self.data.Low[trough_indices[-1]], self.ema_200[-1], self.ema_800[-1]
                if tp1 > price and sl < price and tp2 > price:
                    self.buy(size=0.5, sl=sl, tp=tp1)
                    self.buy(size=0.5, sl=sl, tp=tp2)
            elif is_m and crossover(40, self.rsi) and price < self.ema_50[-1] and retest_short:
                sl, tp1, tp2 = self.data.High[peak_indices[-1]], self.ema_200[-1], self.ema_800[-1]
                if tp1 < price and sl > price and tp2 < price:
                    self.sell(size=0.5, sl=sl, tp=tp1)
                    self.sell(size=0.5, sl=sl, tp=tp2)

if __name__ == '__main__':
    data = GOOG
    bt = Backtest(data, ReversalWM50EmaRsiStrategy, cash=10000, commission=.002)
    stats = bt.optimize(rsi_period=range(10, 30, 5), ema_50_len=range(40, 60, 5),
                        distance=range(3, 10, 2), maximize='Sharpe Ratio')

    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'reversal_w_m_50ema_rsi',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    bt.plot()
