import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from enum import Enum
import json
from scipy.signal import find_peaks

# =====================================================================================
# Helper Functions
# =====================================================================================

def fvg_indicator(high: np.ndarray, low: np.ndarray):
    """
    Identifies Fair Value Gaps (FVGs).
    The FVG is marked at the index of the middle candle of the 3-candle pattern.
    """
    bullish_fvg_top = np.full_like(high, np.nan)
    bullish_fvg_bottom = np.full_like(high, np.nan)
    bearish_fvg_top = np.full_like(high, np.nan)
    bearish_fvg_bottom = np.full_like(high, np.nan)

    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    high_next = np.roll(high, -1)
    low_next = np.roll(low, -1)

    # Bullish FVG: High of candle (i-1) is lower than the Low of candle (i+1)
    bullish_mask = high_prev < low_next
    bullish_fvg_top[bullish_mask] = low_next[bullish_mask]
    bullish_fvg_bottom[bullish_mask] = high_prev[bullish_mask]

    # Bearish FVG: Low of candle (i-1) is higher than the High of candle (i+1)
    bearish_mask = low_prev > high_next
    bearish_fvg_top[bearish_mask] = low_prev[bearish_mask]
    bearish_fvg_bottom[bearish_mask] = high_next[bearish_mask]

    # Avoid wraparound issues from np.roll
    for arr in [bullish_fvg_top, bullish_fvg_bottom, bearish_fvg_top, bearish_fvg_bottom]:
        arr[0] = arr[-1] = np.nan

    return bullish_fvg_top, bullish_fvg_bottom, bearish_fvg_top, bearish_fvg_bottom

def swing_indicator(price: np.ndarray, distance: int):
    """
    Identifies swing highs and lows using scipy.signal.find_peaks.
    """
    peak_indices, _ = find_peaks(price, distance=distance)
    trough_indices, _ = find_peaks(-price, distance=distance)

    swing_highs = np.full_like(price, False, dtype=bool)
    swing_lows = np.full_like(price, False, dtype=bool)

    swing_highs[peak_indices] = True
    swing_lows[trough_indices] = True

    return swing_highs, swing_lows

def ema_indicator(series, period):
    """Calculates the Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, adjust=False).mean()

# =====================================================================================
# State Management
# =====================================================================================

class StrategyState(Enum):
    SCANNING_FOR_HTF_LEVEL_INTERACTION = 1
    WAITING_FOR_LIQUIDITY_SWEEP = 2
    WAITING_FOR_LTF_CONFIRMATION = 3 # MSS + FVG
    WAITING_FOR_ENTRY = 4

# =====================================================================================
# Main Strategy Class
# =====================================================================================

class FourStepFrameworkStrategy(Strategy):
    # Strategy Parameters
    ema_period = 200
    swing_distance_minor = 10
    swing_distance_major = 50
    min_rr = 3.0
    proximity_pct = 0.01
    invalidation_pct = 0.01

    def init(self):
        # 1. PROXY FOR HIGHER-TIMEFRAME DATA & INDICATORS

        # Trend Proxy
        self.ema_long = self.I(ema_indicator, self.data.Close, self.ema_period)

        # Pre-calculate daily levels and merge them back
        df = self.data.df.copy()

        # Ensure timezone-aware index, then convert to naive for resampling
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        daily_agg = df.resample('D').agg({
            'High': 'max',
            'Low': 'min'
        })
        daily_agg['prev_day_high'] = daily_agg['High'].shift(1)
        daily_agg['prev_day_low'] = daily_agg['Low'].shift(1)

        # Merge back into the main DataFrame
        df['date'] = df.index.date

        # Convert daily_agg index to date objects for merging
        daily_agg.index = daily_agg.index.date

        merged_df = pd.merge(df, daily_agg[['prev_day_high', 'prev_day_low']],
                             left_on='date', right_index=True, how='left')

        # Use backfill and forward-fill to handle weekends/missing data
        merged_df['prev_day_high'] = merged_df['prev_day_high'].bfill().ffill()
        merged_df['prev_day_low'] = merged_df['prev_day_low'].bfill().ffill()

        # Add these as accessible data columns for the strategy
        self.data.df['prev_day_high'] = merged_df['prev_day_high']
        self.data.df['prev_day_low'] = merged_df['prev_day_low']

        # Other indicators
        self.bullish_fvg_top, self.bullish_fvg_bottom, \
        self.bearish_fvg_top, self.bearish_fvg_bottom = self.I(
            fvg_indicator, self.data.High, self.data.Low, name="FVG"
        )
        self.swing_highs, self.swing_lows = self.I(
            swing_indicator, self.data.High, self.swing_distance_minor
        )
        self.major_swing_highs, self.major_swing_lows = self.I(
            swing_indicator, self.data.High, self.swing_distance_major
        )

        # 2. STATE MANAGEMENT INITIALIZATION
        self.state = StrategyState.SCANNING_FOR_HTF_LEVEL_INTERACTION
        self.setup_details = {} # To store details of the current trade setup

    def next(self):
        # Always check for exits first
        if self.position:
            return

        # Reset state if a setup becomes invalid or is completed
        if self.state != StrategyState.SCANNING_FOR_HTF_LEVEL_INTERACTION and not self._is_setup_still_valid():
            self._reset_state()

        # =============================================================================
        # STATE MACHINE LOGIC
        # =============================================================================

        if self.state == StrategyState.SCANNING_FOR_HTF_LEVEL_INTERACTION:
            self._handle_scanning()

        elif self.state == StrategyState.WAITING_FOR_LIQUIDITY_SWEEP:
            self._handle_liquidity_sweep()

        elif self.state == StrategyState.WAITING_FOR_LTF_CONFIRMATION:
            self._handle_ltf_confirmation()

        elif self.state == StrategyState.WAITING_FOR_ENTRY:
            self._handle_entry()

    def _reset_state(self):
        self.state = StrategyState.SCANNING_FOR_HTF_LEVEL_INTERACTION
        self.setup_details = {}

    def _is_setup_still_valid(self):
        """Check if the current setup should be invalidated."""
        if not self.setup_details:
            return False

        # Invalidate if price moves too far from the initial sweep level
        invalidation_threshold = self.setup_details.get('invalidation_price')
        if invalidation_threshold is None:
            return True # No invalidation level set yet

        if self.setup_details['direction'] == 'short' and self.data.High[-1] > invalidation_threshold:
            return False
        if self.setup_details['direction'] == 'long' and self.data.Low[-1] < invalidation_threshold:
            return False

        return True

    def _handle_scanning(self):
        """STEP 1 & 2: Find interaction with a daily key level with directional bias."""
        is_uptrend = self.data.Close[-1] > self.ema_long[-1]
        is_downtrend = self.data.Close[-1] < self.ema_long[-1]

        prev_day_high = self.data.df['prev_day_high'].iloc[-1]
        prev_day_low = self.data.df['prev_day_low'].iloc[-1]

        # Look for a short setup: downtrend + price near previous day's high
        if is_downtrend and abs(self.data.High[-1] - prev_day_high) / prev_day_high < self.proximity_pct:
            self.state = StrategyState.WAITING_FOR_LIQUIDITY_SWEEP
            self.setup_details = {
                'direction': 'short',
                'key_level': prev_day_high,
                'invalidation_price': prev_day_high * (1 + self.invalidation_pct)
            }

        # Look for a long setup: uptrend + price near previous day's low
        elif is_uptrend and abs(self.data.Low[-1] - prev_day_low) / prev_day_low < self.proximity_pct:
            self.state = StrategyState.WAITING_FOR_LIQUIDITY_SWEEP
            self.setup_details = {
                'direction': 'long',
                'key_level': prev_day_low,
                'invalidation_price': prev_day_low * (1 - self.invalidation_pct)
            }

    def _handle_liquidity_sweep(self):
        """STEP 3: Confirm a sweep of the identified key level."""
        direction = self.setup_details['direction']
        key_level = self.setup_details['key_level']

        # For a short, price must trade above the key level
        if direction == 'short' and self.data.High[-1] > key_level:
            self.state = StrategyState.WAITING_FOR_LTF_CONFIRMATION
            self.setup_details['sweep_high'] = self.data.High[-1]

        # For a long, price must trade below the key level
        elif direction == 'long' and self.data.Low[-1] < key_level:
            self.state = StrategyState.WAITING_FOR_LTF_CONFIRMATION
            self.setup_details['sweep_low'] = self.data.Low[-1]

    def _handle_ltf_confirmation(self):
        """STEP 4 (Part 1): Look for Market Structure Shift (MSS) + FVG."""
        direction = self.setup_details['direction']

        # For a short, we need a break of a recent swing low
        if direction == 'short':
            recent_swing_lows = np.where(self.swing_lows)[0]
            if recent_swing_lows.any():
                last_swing_low_idx = recent_swing_lows[-1]
                last_swing_low_price = self.data.Low[last_swing_low_idx]

                # Check for MSS: Close price breaks below the last swing low
                if self.data.Close[-1] < last_swing_low_price:
                    self.setup_details['mss_confirmed'] = True

            # After MSS, look for a new bearish FVG
            if self.setup_details.get('mss_confirmed') and not np.isnan(self.bearish_fvg_top[-2]):
                self.state = StrategyState.WAITING_FOR_ENTRY
                self.setup_details['fvg_top'] = self.bearish_fvg_top[-2]
                self.setup_details['fvg_bottom'] = self.bearish_fvg_bottom[-2]

        # For a long, we need a break of a recent swing high
        elif direction == 'long':
            recent_swing_highs = np.where(self.swing_highs)[0]
            if recent_swing_highs.any():
                last_swing_high_idx = recent_swing_highs[-1]
                last_swing_high_price = self.data.High[last_swing_high_idx]

                # Check for MSS: Close price breaks above the last swing high
                if self.data.Close[-1] > last_swing_high_price:
                    self.setup_details['mss_confirmed'] = True

            # After MSS, look for a new bullish FVG
            if self.setup_details.get('mss_confirmed') and not np.isnan(self.bullish_fvg_top[-2]):
                self.state = StrategyState.WAITING_FOR_ENTRY
                self.setup_details['fvg_top'] = self.bullish_fvg_top[-2]
                self.setup_details['fvg_bottom'] = self.bullish_fvg_bottom[-2]

    def _handle_entry(self):
        """STEP 4 (Part 2): Enter on retracement into the FVG and manage risk."""
        direction = self.setup_details['direction']
        fvg_top = self.setup_details['fvg_top']
        fvg_bottom = self.setup_details['fvg_bottom']

        # Short Entry: Price touches the bottom of a bearish FVG
        if direction == 'short' and self.data.High[-1] >= fvg_bottom:
            entry_price = self.data.Close[-1]
            sl = self.setup_details['sweep_high']

            # Find a major swing low as a target
            major_swing_lows_indices = np.where(self.major_swing_lows)[0]
            valid_targets = self.data.Low[major_swing_lows_indices][self.data.Low[major_swing_lows_indices] < entry_price]
            if not valid_targets.any():
                self._reset_state()
                return

            tp = np.max(valid_targets) # Closest major swing low

            # RR check
            if (entry_price - sl) != 0 and (tp - entry_price) / (entry_price - sl) >= self.min_rr:
                self.sell(sl=sl, tp=tp)
            self._reset_state()

        # Long Entry: Price touches the top of a bullish FVG
        elif direction == 'long' and self.data.Low[-1] <= fvg_top:
            entry_price = self.data.Close[-1]
            sl = self.setup_details['sweep_low']

            # Find a major swing high as a target
            major_swing_highs_indices = np.where(self.major_swing_highs)[0]
            valid_targets = self.data.High[major_swing_highs_indices][self.data.High[major_swing_highs_indices] > entry_price]
            if not valid_targets.any():
                self._reset_state()
                return

            tp = np.min(valid_targets) # Closest major swing high

            # RR check
            if (sl - entry_price) != 0 and (tp - entry_price) / (entry_price - sl) >= self.min_rr:
                self.buy(sl=sl, tp=tp)
            self._reset_state()

# =====================================================================================
# Backtest Execution
# =====================================================================================

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    try:
        # Load data more robustly, specifying column names
        data = pd.read_csv(
            data_path,
            index_col='datetime',
            parse_dates=True,
            header=0,
            names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unnamed'],
            usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        exit()

    bt = Backtest(data, FourStepFrameworkStrategy, cash=100_000, commission=.002)
    stats = bt.run()

    print(stats)

    # Save stats to a JSON file
    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON output
    sanitized_stats = {key: (val if pd.api.types.is_number(val) else str(val)) for key, val in stats.items()}

    result = {
        'strategy_name': 'four_step_framework',
        'return': sanitized_stats.get('Return [%]', 0),
        'sharpe': sanitized_stats.get('Sharpe Ratio', 0),
        'max_drawdown': sanitized_stats.get('Max. Drawdown [%]', 0),
        'win_rate': sanitized_stats.get('Win Rate [%]', 0),
        'total_trades': sanitized_stats.get('# Trades', 0)
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result, f, indent=4)

    # Generate plot
    plot_filename = 'results/four_step_framework.html'
    try:
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
