from backtesting import Strategy, Backtest
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import json
import os
import pandas_ta as ta

def sanitize_stats(stats):
    """Sanitizes the stats dictionary for JSON serialization."""
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            sanitized[key] = None
        elif pd.isna(value):
            sanitized[key] = None
        elif isinstance(value, (np.int64, np.int32)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            sanitized[key] = float(value)
        else:
            sanitized[key] = value
    return sanitized

def atr_indicator(high, low, close, length):
    """Wrapper for pandas-ta ATR to be used with backtesting.py."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    atr = ta.atr(high=high, low=low, close=close, length=length)
    if atr is None:
        return np.full(len(high), np.nan)
    return atr.values


class ElliottWaveFifthWaveExtensionReversal(Strategy):
    """
    Strategy to identify and trade the reversal after a fifth wave extension
    in an Elliott Wave impulse pattern.
    """
    # Parameters for wave detection
    peak_distance = 20  # How far apart peaks/troughs should be
    wave_equality_tolerance = 0.20  # Tolerance for comparing wave lengths (20%)
    extension_multiplier = 1.618  # Threshold for identifying a 5th wave extension

    # Parameters for ATR-based Stop Loss
    atr_period = 14
    atr_multiplier = 2.0

    def init(self):
        """
        Initialize the strategy.
        """
        # Using Close prices for wave analysis
        self.price = self.data.Close
        self.volume = self.data.Volume
        # Calculate ATR
        self.atr = self.I(atr_indicator, self.data.High, self.data.Low, self.data.Close, length=self.atr_period)

    def _find_wave_points(self, data):
        """Finds peaks and troughs in the given data series."""
        peaks, _ = find_peaks(data, distance=self.peak_distance)
        troughs, _ = find_peaks(-data, distance=self.peak_distance)

        # Combine and sort all extrema
        extrema = np.sort(np.concatenate([peaks, troughs]))
        return extrema

    def _find_last_impulse_wave(self):
        """
        Scans backwards from the current bar to find the last valid 5-point impulse wave.
        """
        # Look at the last ~500 bars of data to find waves
        window_size = 500
        if len(self.price) < window_size:
            return None

        recent_prices = self.price[-window_size:]
        extrema_indices = self._find_wave_points(recent_prices)

        # We need at least 5 points to form an impulse wave
        if len(extrema_indices) < 5:
            return None

        # Iterate backwards through extrema to find the most recent valid wave
        for i in range(len(extrema_indices) - 5, -1, -1):
            potential_wave_indices = extrema_indices[i : i + 5]

            # Adjust indices to be relative to the full dataset
            absolute_indices = len(self.price) - window_size + potential_wave_indices

        if len(extrema_indices) < 6:
            return None

        for i in range(len(extrema_indices) - 6, -1, -1):
            wave_cand_indices = extrema_indices[i : i + 6]
            abs_indices = len(self.price) - window_size + wave_cand_indices

            prices = self.price[abs_indices]
            p0, p1, p2, p3, p4, p5 = prices

            # Uptrend impulse wave: T-P-T-P-T-P
            is_uptrend_structure = p0 < p1 and p2 < p1 and p2 < p3 and p4 < p3 and p4 < p5
            if not is_uptrend_structure:
                continue

            # Elliott Wave Rules Validation
            wave1_len = p1 - p0
            wave2_retracement = (p1 - p2) / wave1_len
            wave3_len = p3 - p2
            wave4_retracement = (p3 - p4) / wave3_len
            wave5_len = p5 - p4

            # Rule 1: Wave 2 doesn't retrace more than 100% of wave 1
            if wave2_retracement >= 1.0:
                continue

            # Rule 2: Wave 3 is not the shortest wave
            if wave3_len < wave1_len and wave3_len < wave5_len:
                continue

            # Rule 3: Wave 4 does not overlap with wave 1's price territory
            if p4 <= p1:
                continue

            # Found a valid impulse wave. Now check for 5th wave extension.
            # Heuristic: Wave 1 and 3 are approx equal
            if abs(wave1_len - wave3_len) / max(wave1_len, wave3_len) < self.wave_equality_tolerance:
                # And wave 5 is extended
                if wave5_len > wave3_len * self.extension_multiplier:
                    # Volume confirmation for the 5th wave
                    vol_w3_start, vol_w3_end = abs_indices[2], abs_indices[3]
                    vol_w5_start, vol_w5_end = abs_indices[4], abs_indices[5]

                    avg_vol_w3 = self.volume[vol_w3_start:vol_w3_end+1].mean()
                    avg_vol_w5 = self.volume[vol_w5_start:vol_w5_end+1].mean()

                    if avg_vol_w5 > avg_vol_w3:
                        # This is our setup!
                        # Subwave analysis for take-profit target
                        wave5_start_idx, wave5_end_idx = abs_indices[4], abs_indices[5]
                        wave5_prices = self.price[wave5_start_idx : wave5_end_idx + 1]

                        # Find sub-extrema within wave 5. Use a smaller distance.
                        sub_peak_dist = max(2, int(self.peak_distance / 4))
                        sub_extrema, _ = find_peaks(wave5_prices, distance=sub_peak_dist)
                        sub_troughs, _ = find_peaks(-wave5_prices, distance=sub_peak_dist)

                        all_sub_extrema = np.sort(np.concatenate([sub_extrema, sub_troughs]))

                        # The structure of an extended 5th wave is 5-3-5-3-5... total 9 subwaves
                        # For an uptrend, this would be P-T-P-T-P-T-P-T-P
                        # We need the low of wave 2 of the extension.
                        # The first trough within the 5th wave is the low of subwave 2.
                        if len(sub_troughs) > 0:
                            # The first trough is the low of the second sub-wave
                            low_of_subwave2_idx = wave5_start_idx + sub_troughs[0]
                            take_profit_price = self.price[low_of_subwave2_idx]

                            atr_at_peak = self.atr[abs_indices[5]]
                            stop_loss_price = p5 + (atr_at_peak * self.atr_multiplier)

                            wave_info = {
                                "indices": abs_indices,
                                "prices": prices,
                                "wave5_peak_idx": abs_indices[5],
                                "stop_loss": stop_loss_price,
                                "take_profit": take_profit_price
                            }
                            return wave_info
        return None


    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        # Only check for a new pattern if we are not in a position
        if self.position:
            return

        # We look for a reversal at the end of the current bar, so we need enough history
        if len(self.data) < 500:
            return

        # Check if the last bar is a peak
        current_price = self.price[-1]
        is_potential_peak = len(self.price) > self.peak_distance * 2 and \
                            all(current_price > self.price[-(i+2)] for i in range(self.peak_distance))

        if not is_potential_peak:
            return

        wave_setup = self._find_last_impulse_wave()

        if wave_setup:
            # Check if the identified peak is the current bar
            if wave_setup['wave5_peak_idx'] == len(self.price) - 1:
                 # We have a setup. Enter short.
                sl = wave_setup['stop_loss']
                tp = wave_setup['take_profit']

                if tp < self.price[-1]: # Ensure TP is below current price for a short
                    self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    data_path = 'data/crypto/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # Load data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Clean up column names
    data.columns = [c.strip().capitalize() for c in data.columns]

    # Ensure OHLC format
    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        raise ValueError("Data must contain Open, High, Low, Close, Volume columns")

    # Run backtest
    bt = Backtest(data, ElliottWaveFifthWaveExtensionReversal, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()
    print(stats)

    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    result_data = {
        'strategy_name': 'elliott_wave_fifth_wave_extension_reversal',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    # Sanitize and save
    clean_stats = sanitize_stats(result_data)
    with open(os.path.join(results_dir, 'temp_result.json'), 'w') as f:
        json.dump(clean_stats, f, indent=2)
        f.write('\n')

    print(f"Backtest results saved to {os.path.join(results_dir, 'temp_result.json')}")

    # Generate plot
    plot_filename = os.path.join(results_dir, 'elliott_wave_fifth_wave_extension_reversal.html')
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
