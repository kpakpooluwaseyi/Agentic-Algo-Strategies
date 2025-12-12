
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks

# Custom indicator function to find swing points
def find_swing_points(series, distance):
    # find_peaks from scipy returns peaks (swing highs)
    peaks, _ = find_peaks(series, distance=distance)
    # To find troughs (swing lows), we invert the series
    troughs, _ = find_peaks(-series, distance=distance)
    return peaks, troughs

class CorrectiveWaveFadeWithVolumeDivergenceStrategy(Strategy):
    # --- Strategy Parameters ---
    # Lookback period for identifying swing points
    swing_lookback = 10
    # Small buffer for stop loss placement
    sl_buffer_percent = 0.01

    def init(self):
        # --- State Machine ---
        # SEARCHING_MAJOR_LOW: Looking for the initial low before the corrective wave.
        # SEARCHING_PHH: Found the major low, now looking for the Previous Higher High.
        # SEARCHING_HL: Found the PHH, now looking for the Higher Low.
        # MONITORING_ENTRY: Found the HL, waiting for a break below to enter.
        self.state = 'SEARCHING_MAJOR_LOW'

        # --- Pattern Points ---
        self.major_low = None
        self.phh = None  # Previous Higher High
        self.hl = None   # Higher Low

        # --- Pre-calculate swing points for efficiency ---
        self.highs = self.data.High
        self.lows = self.data.Low
        self.peak_indices, self.trough_indices = find_swing_points(self.highs, self.swing_lookback)

        # Convert to sets for faster lookups in next()
        self.peak_indices = set(self.peak_indices)
        self.trough_indices = set(self.trough_indices)

    def next(self):
        current_index = len(self.data.Close) - 1

        # =====================================================================
        # STATE 1: SEARCHING FOR THE INITIAL MAJOR LOW
        # =====================================================================
        if self.state == 'SEARCHING_MAJOR_LOW':
            if current_index in self.trough_indices:
                self.major_low = (current_index, self.lows[current_index])
                self.state = 'SEARCHING_PHH'

        # =====================================================================
        # STATE 2: SEARCHING FOR THE PREVIOUS HIGHER HIGH (PHH)
        # =====================================================================
        elif self.state == 'SEARCHING_PHH':
            # Invalidation: If price makes a new low below our starting point
            if self.lows[current_index] < self.major_low[1]:
                self.state = 'SEARCHING_MAJOR_LOW'
                return

            if current_index in self.peak_indices and current_index > self.major_low[0]:
                self.phh = (current_index, self.highs[current_index])

                # --- Volume Divergence Check ---
                start_idx = self.major_low[0]
                end_idx = self.phh[0]

                if end_idx - start_idx > 2: # Need at least 3 bars to check trend
                    volume_slice = self.data.Volume[start_idx:end_idx+1]
                    time_slice = np.arange(len(volume_slice))

                    # Fit a line to the volume data; slope indicates the trend
                    slope, _ = np.polyfit(time_slice, volume_slice, 1)

                    # If volume is declining (negative slope), the condition is met
                    if slope < 0:
                        self.state = 'SEARCHING_HL'
                    else: # Volume not declining, reset
                        self.state = 'SEARCHING_MAJOR_LOW'
                else: # Not enough data, reset
                    self.state = 'SEARCHING_MAJOR_LOW'

        # =====================================================================
        # STATE 3: SEARCHING FOR THE HIGHER LOW (HL)
        # =====================================================================
        elif self.state == 'SEARCHING_HL':
            # Invalidation: If price makes a new high above PHH
            if self.highs[current_index] > self.phh[1]:
                self.state = 'SEARCHING_MAJOR_LOW'
                return

            if current_index in self.trough_indices and current_index > self.phh[0]:
                hl_candidate_price = self.lows[current_index]

                # --- Overlap Condition Check ---
                # The HL must be lower than the PHH but higher than the Major Low.
                if self.major_low[1] < hl_candidate_price < self.phh[1]:
                    self.hl = (current_index, hl_candidate_price)
                    self.state = 'MONITORING_ENTRY'

        # =====================================================================
        # STATE 4: MONITORING FOR ENTRY TRIGGER
        # =====================================================================
        elif self.state == 'MONITORING_ENTRY':
            # Invalidation: If price makes a new high above PHH
            if self.highs[current_index] > self.phh[1]:
                self.state = 'SEARCHING_MAJOR_LOW'
                return

            # --- Entry Trigger ---
            # If the current candle's close breaks below the established HL.
            if self.data.Close[-1] < self.hl[1]:
                # Define SL and TP
                stop_loss = self.phh[1] * (1 + self.sl_buffer_percent)
                take_profit = self.major_low[1]

                # --- Pre-trade Validation ---
                # Ensure the TP is valid relative to the current price (proxy for entry)
                if take_profit < self.data.Close[-1]:
                    # Place the short trade
                    self.sell(sl=stop_loss, tp=take_profit)

                # Reset state after the entry condition is met, regardless of whether
                # the trade was placed. The setup is considered complete/invalidated.
                self.state = 'SEARCHING_MAJOR_LOW'

def generate_synthetic_data(num_candles=1000):
    """
    Generates synthetic OHLCV data with a clear corrective wave fade pattern.
    """
    np.random.seed(42)
    time_index = pd.date_range(start='2023-01-01', periods=num_candles, freq='4H')
    price = np.zeros(num_candles)
    volume = np.zeros(num_candles)

    # Baseline random walk
    price[0] = 100
    price[1:] = 100 + np.random.randn(num_candles - 1).cumsum() * 0.2
    volume = np.random.randint(500, 1500, num_candles)

    # --- Inject the specific pattern ---
    # Major Low (index 200)
    price[200] = 80

    # Corrective Rise (indices 201 to 250)
    # This rise will form the PHH and the HL
    start_price = price[200]

    # PHH (Previous Higher High) at index 225
    phh_price = start_price + 15
    price[201:226] = np.linspace(start_price, phh_price, 25)

    # HL (Higher Low) at index 240
    hl_price = start_price + 5
    price[226:241] = np.linspace(phh_price, hl_price, 15)

    # Subsequent move before breakdown
    price[241:250] = np.linspace(hl_price, hl_price + 2, 9)

    # Breakdown below HL (at index 251)
    price[251] = hl_price - 1
    price[252:300] = price[251] - np.random.rand(48).cumsum()

    # --- Volume Divergence ---
    # Volume should decrease during the rise from Major Low to PHH
    volume[201:226] = np.linspace(2000, 800, 25).astype(int)

    data = pd.DataFrame(index=time_index)
    data['Open'] = price
    data['High'] = data['Open'] + np.random.uniform(0, 0.5, num_candles)
    data['Low'] = data['Open'] - np.random.uniform(0, 0.5, num_candles)
    data['Close'] = data['Open'] + np.random.uniform(-0.2, 0.2, num_candles)
    data['Volume'] = volume

    # Ensure OHLC consistency and place specific pattern points
    data.at[data.index[200], 'Low'] = 80
    data.at[data.index[225], 'High'] = phh_price
    data.at[data.index[240], 'Low'] = hl_price
    data.at[data.index[251], 'Close'] = hl_price - 1

    return data

if __name__ == '__main__':
    # --- Generate Synthetic Data ---
    data = generate_synthetic_data(num_candles=2000)

    # --- Initialize Backtest ---
    bt = Backtest(data, CorrectiveWaveFadeWithVolumeDivergenceStrategy, cash=100_000, commission=.002)

    # --- Optimization ---
    stats = bt.optimize(
        swing_lookback=range(5, 21, 5),
        sl_buffer_percent=np.arange(0.005, 0.021, 0.005).tolist(),
        maximize='Sharpe Ratio'
    )

    print("Best run stats:")
    print(stats)

    # --- Save Results ---
    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize results for JSON serialization
    results_dict = {
        'strategy_name': 'corrective_wave_fade_with_volume_divergence',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
        f.write('\n')

    # --- Generate Plot ---
    print("Generating plot...")
    try:
        bt.plot(filename='results/corrective_wave_fade.html', open_browser=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")
