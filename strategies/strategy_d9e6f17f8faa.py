import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import json
from scipy.signal import find_peaks

# --- Indicator Helper Functions ---

def swing_indicator(price: np.ndarray, distance: int):
    """
    Identifies swing highs and lows using scipy.signal.find_peaks.
    Returns boolean arrays marking the locations of swing points.
    """
    peak_indices, _ = find_peaks(price, distance=distance)
    trough_indices, _ = find_peaks(-price, distance=distance)
    swing_highs = np.full_like(price, False, dtype=bool)
    swing_lows = np.full_like(price, False, dtype=bool)
    swing_highs[peak_indices] = True
    swing_lows[trough_indices] = True
    return swing_highs, swing_lows

def fvg_indicator(high: np.ndarray, low: np.ndarray):
    """
    Identifies Fair Value Gaps (FVGs) using vectorized NumPy operations.
    """
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    high_next = np.roll(high, -1)
    low_next = np.roll(low, -1)

    bullish_mask = (high_prev < low_next)
    bearish_mask = (low_prev > high_next)

    bullish_fvg_top = np.where(bullish_mask, low_next, np.nan)
    bullish_fvg_bottom = np.where(bullish_mask, high_prev, np.nan)
    bearish_fvg_top = np.where(bearish_mask, low_prev, np.nan)
    bearish_fvg_bottom = np.where(bearish_mask, high_next, np.nan)

    for arr in [bullish_fvg_top, bullish_fvg_bottom, bearish_fvg_top, bearish_fvg_bottom]:
        arr[0] = arr[-1] = np.nan

    return bullish_fvg_top, bullish_fvg_bottom, bearish_fvg_top, bearish_fvg_bottom

# --- Data Pre-processing ---

def preprocess_data(df):
    """
    Calculates higher timeframe key levels (PDH/L, PWH/L) and adds them to the DataFrame.
    """
    df.index = pd.to_datetime(df.index)

    # Calculate Previous Day High/Low
    daily_high = df['High'].resample('D').max()
    daily_low = df['Low'].resample('D').min()
    df['PDH'] = daily_high.shift(1).reindex(df.index, method='ffill')
    df['PDL'] = daily_low.shift(1).reindex(df.index, method='ffill')

    # Calculate Previous Week High/Low
    weekly_high = df['High'].resample('W').max()
    weekly_low = df['Low'].resample('W').min()
    df['PWH'] = weekly_high.shift(1).reindex(df.index, method='ffill')
    df['PWL'] = weekly_low.shift(1).reindex(df.index, method='ffill')

    # df.dropna(inplace=True) # Avoid dropping all data
    return df

# --- Strategy Class ---

class PredictableCandleExpansion(Strategy):
    # --- Strategy Parameters ---
    swing_distance = 10
    min_rr = 3.0
    sl_buffer_pct = 0.1

    # --- State Machine ---
    STATE_SCANNING = 0
    STATE_AWAITING_DISPLACEMENT = 1
    STATE_AWAITING_ENTRY = 2

    def init(self):
        # --- Indicators ---
        self.swing_highs, self.swing_lows = self.I(
            swing_indicator, self.data.Close, self.swing_distance
        )
        self.bullish_fvg_top, self.bullish_fvg_bottom, \
        self.bearish_fvg_top, self.bearish_fvg_bottom = self.I(
            fvg_indicator, self.data.High, self.data.Low, name="FVG"
        )
        # Access pre-computed levels
        self.pdh = self.data.PDH
        self.pdl = self.data.PDL

        # --- State Management ---
        self.state = self.STATE_SCANNING
        self.bias = 0  # 1 for bullish, -1 for bearish
        self.stop_run_level = None
        self.displacement_candle_idx = None
        self.active_fvg = None

    def next(self):
        # --- Data Integrity Check ---
        if pd.isna(self.pdh[-1]) or pd.isna(self.pdl[-1]):
            return

        # --- Exit Logic ---
        if self.position:
            # Simple exit: hold until stopped out or TP is hit
            return

        # --- State Machine Logic ---
        if self.state == self.STATE_SCANNING:
            self._handle_scanning_state()
        elif self.state == self.STATE_AWAITING_DISPLACEMENT:
            self._handle_awaiting_displacement_state()
        elif self.state == self.STATE_AWAITING_ENTRY:
            self._handle_awaiting_entry_state()

    def _handle_scanning_state(self):
        """
        State 1: Scan for a higher timeframe bias and a subsequent stop run.
        """
        # Determine bias based on PDH/PDL breach
        if self.data.High[-1] > self.pdh[-1]:
            self.bias = 1
        elif self.data.Low[-1] < self.pdl[-1]:
            self.bias = -1
        else:
            self.bias = 0 # No clear bias if inside previous day's range

        if self.bias == 0:
            return

        # Look for a stop run (liquidity grab)
        # For a bullish bias, we expect a run on a recent swing low
        if self.bias == 1 and self.swing_lows[-2]:
            self.stop_run_level = self.data.Low[-2]
            self.state = self.STATE_AWAITING_DISPLACEMENT

        # For a bearish bias, we expect a run on a recent swing high
        elif self.bias == -1 and self.swing_highs[-2]:
            self.stop_run_level = self.data.High[-2]
            self.state = self.STATE_AWAITING_DISPLACEMENT

    def _handle_awaiting_displacement_state(self):
        """
        State 2: After a stop run, wait for a displacement candle that creates an FVG.
        """
        # Invalidate if price moves too far against the intended direction
        if self.bias == 1 and self.data.Low[-1] < self.stop_run_level:
            self._reset_state()
            return
        if self.bias == -1 and self.data.High[-1] > self.stop_run_level:
            self._reset_state()
            return

        # Check for FVG formation after the stop run
        if self.bias == 1 and not np.isnan(self.bullish_fvg_top[-2]):
            self.active_fvg = ('bullish', self.bullish_fvg_top[-2], self.bullish_fvg_bottom[-2])
            self.displacement_candle_idx = len(self.data) - 2
            self.state = self.STATE_AWAITING_ENTRY

        elif self.bias == -1 and not np.isnan(self.bearish_fvg_top[-2]):
            self.active_fvg = ('bearish', self.bearish_fvg_top[-2], self.bearish_fvg_bottom[-2])
            self.displacement_candle_idx = len(self.data) - 2
            self.state = self.STATE_AWAITING_ENTRY

    def _handle_awaiting_entry_state(self):
        """
        State 3: FVG is identified. Wait for price to retrace into it for an entry.
        """
        fvg_type, fvg_top, fvg_bottom = self.active_fvg

        # Invalidate if a new candle closes beyond the FVG, negating it
        if (fvg_type == 'bullish' and self.data.Close[-1] < fvg_bottom) or \
           (fvg_type == 'bearish' and self.data.Close[-1] > fvg_top):
            self._reset_state()
            return

        entry_price = None
        # Check for retracement into the FVG
        if fvg_type == 'bullish' and self.data.Low[-1] <= fvg_top:
            entry_price = self.data.Close[-1]
            sl = self.stop_run_level * (1 - self.sl_buffer_pct / 100)
            tp = self.pdh[-1] # Target next major liquidity level

            # RR Check
            if (tp - entry_price) / (entry_price - sl) >= self.min_rr:
                self.buy(sl=sl, tp=tp)
            else:
                self._reset_state()

        elif fvg_type == 'bearish' and self.data.High[-1] >= fvg_bottom:
            entry_price = self.data.Close[-1]
            sl = self.stop_run_level * (1 + self.sl_buffer_pct / 100)
            tp = self.pdl[-1] # Target next major liquidity level

            # RR Check
            if (entry_price - tp) / (sl - entry_price) >= self.min_rr:
                self.sell(sl=sl, tp=tp)
            else:
                self._reset_state()

    def _reset_state(self):
        """Resets the state machine to its initial state."""
        self.state = self.STATE_SCANNING
        self.bias = 0
        self.stop_run_level = None
        self.displacement_candle_idx = None
        self.active_fvg = None

# --- Backtest Execution ---

if __name__ == '__main__':
    import os

    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    strategy_name = 'predictable_candle_expansion'

    # --- Data Loading and Preprocessing ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Sanitize column names to handle potential whitespace issues
    data.columns = [c.strip().title() for c in data.columns]

    print("Preprocessing data...")
    data = preprocess_data(data.copy())

    # --- Backtesting ---
    print("Running backtest...")
    bt = Backtest(data, PredictableCandleExpansion, cash=100_000, commission=.002)
    stats = bt.run()

    # --- Results ---
    print(stats)

    os.makedirs('results', exist_ok=True)
    plot_filename = f'results/{strategy_name}_plot.html'
    results_filename = 'results/temp_result.json'

    # Generate plot
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    # Sanitize and save stats
    results_dict = {
        'strategy_name': strategy_name,
        '# Trades': stats.get('# Trades', 0),
        'Return [%]': stats.get('Return [%]', 0.0),
        'Win Rate [%]': stats.get('Win Rate [%]', 0.0),
        'Max. Drawdown [%]': stats.get('Max. Drawdown [%]', 0.0),
        'Sharpe Ratio': stats.get('Sharpe Ratio', 0.0),
    }
    # Ensure all values are JSON serializable
    for key, value in results_dict.items():
        if isinstance(value, (np.integer, np.int64)):
            results_dict[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            results_dict[key] = float(value)
        elif pd.isna(value):
            results_dict[key] = None

    with open(results_filename, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {results_filename}")
