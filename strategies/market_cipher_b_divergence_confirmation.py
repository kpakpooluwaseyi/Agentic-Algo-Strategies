import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from scipy.signal import find_peaks
import json
import os
from typing import List, Tuple, Optional

def market_cipher_b_indicator(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    macd_fast=12, macd_slow=26, macd_signal=9, mfi_period=14
):
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    volume_series = pd.Series(volume)

    macd_df = ta.macd(close_series, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    macd_line = macd_df[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}']
    signal_line = macd_df[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}']

    mfi = ta.mfi(high_series, low_series, close_series, volume_series, length=mfi_period)

    return macd_line.values, signal_line.values, mfi.values

class MarketCipherBDivergenceConfirmation(Strategy):
    divergence_lookback = 100
    stop_loss_buffer_pct = 0.01
    swing_proximity_threshold = 10
    mfi_exit_long = 40
    mfi_exit_short = 60

    def init(self):
        self.macd, self.macd_signal, self.mfi = self.I(
            market_cipher_b_indicator, self.data.High, self.data.Low, self.data.Close, self.data.Volume
        )

    def _find_swings(self, data: np.ndarray, lookback: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        start_idx = max(0, len(data) - lookback)
        window = data[start_idx:]
        high_peaks_idx, _ = find_peaks(window, distance=5)
        low_peaks_idx, _ = find_peaks(-window, distance=5)
        return ([(start_idx + i, window[i]) for i in high_peaks_idx],
                [(start_idx + i, window[i]) for i in low_peaks_idx])

    def _find_corresponding_swing(self, price_swing_idx: int, indicator_swings: List[Tuple[int, float]]) -> Optional[Tuple[int, float]]:
        return next(((idx, val) for idx, val in reversed(indicator_swings) if abs(idx - price_swing_idx) <= self.swing_proximity_threshold), None)

    def next(self):
        if self.position.is_long:
            if crossover(self.macd_signal, self.macd) or self.mfi[-1] < self.mfi_exit_long:
                self.position.close()
                return
        elif self.position.is_short:
            if crossover(self.macd, self.macd_signal) or self.mfi[-1] > self.mfi_exit_short:
                self.position.close()
                return

        if self.position: return

        price_highs, price_lows = self._find_swings(self.data.Close, self.divergence_lookback)
        macd_highs, macd_lows = self._find_swings(self.macd, self.divergence_lookback)

        # Bearish Divergence
        for i in range(len(price_highs) - 1, 0, -1):
            recent_price_high_idx, recent_price_high = price_highs[i]
            recent_macd_swing = self._find_corresponding_swing(recent_price_high_idx, macd_highs)
            if not recent_macd_swing: continue

            for j in range(i - 1, -1, -1):
                prior_price_high_idx, prior_price_high = price_highs[j]
                prior_macd_swing = self._find_corresponding_swing(prior_price_high_idx, macd_highs)
                if not prior_macd_swing: continue

                if recent_price_high > prior_price_high and recent_macd_swing[1] < prior_macd_swing[1]:
                    if self.mfi[-1] < self.mfi[-2]:
                        stop_loss = self.data.High[recent_price_high_idx] * (1 + self.stop_loss_buffer_pct)
                        if self.data.Close[-1] < stop_loss: self.sell(sl=stop_loss); return
                break
            break

        # Bullish Divergence
        for i in range(len(price_lows) - 1, 0, -1):
            recent_price_low_idx, recent_price_low = price_lows[i]
            recent_macd_swing = self._find_corresponding_swing(recent_price_low_idx, macd_lows)
            if not recent_macd_swing: continue

            for j in range(i - 1, -1, -1):
                prior_price_low_idx, prior_price_low = price_lows[j]
                prior_macd_swing = self._find_corresponding_swing(prior_price_low_idx, macd_lows)
                if not prior_macd_swing: continue

                if recent_price_low < prior_price_low and recent_macd_swing[1] > prior_macd_swing[1]:
                    if self.mfi[-1] > self.mfi[-2]:
                        stop_loss = self.data.Low[recent_price_low_idx] * (1 - self.stop_loss_buffer_pct)
                        if self.data.Close[-1] > stop_loss: self.buy(sl=stop_loss); return
                break
            break

if __name__ == '__main__':
    if not os.path.exists('results'): os.makedirs('results')
    data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    data.columns = [c.strip().capitalize() for c in data.columns]

    bt = Backtest(data, MarketCipherBDivergenceConfirmation, cash=100000, commission=.002)
    stats = bt.run()

    print(stats)

    stats_dict = dict(stats)
    stats_dict.pop('_strategy', None); stats_dict.pop('_equity_curve', None); stats_dict.pop('_trades', None)
    for key, value in list(stats_dict.items()):
        if isinstance(value, pd.Timedelta): stats_dict[key] = str(value)
        elif isinstance(value, (np.integer, np.int64)): stats_dict[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)): stats_dict[key] = float(value)
        elif isinstance(value, pd.Timestamp): stats_dict[key] = value.isoformat()
        elif pd.isna(value): stats_dict[key] = None
    with open('results/temp_result.json', 'w') as f: json.dump(stats_dict, f, indent=4)
    try:
        bt.plot(filename='results/market_cipher_b_divergence_confirmation.html', open_browser=False)
    except Exception as e: print(f"Could not generate plot: {e}")
