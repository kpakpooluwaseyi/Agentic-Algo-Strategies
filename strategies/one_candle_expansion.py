import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, resample_apply
import json


def fvg_indicator(high: np.ndarray, low: np.ndarray):
    """
    Identifies Fair Value Gaps (FVGs) using vectorized NumPy operations.
    The FVG is marked at the index of the middle candle of the 3-candle pattern.
    """
    # Initialize output arrays with NaNs
    bullish_fvg_top = np.full_like(high, np.nan)
    bullish_fvg_bottom = np.full_like(high, np.nan)
    bearish_fvg_top = np.full_like(high, np.nan)
    bearish_fvg_bottom = np.full_like(high, np.nan)

    # Shift arrays to get previous (i-1) and next (i+1) candle values
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    high_next = np.roll(high, -1)
    low_next = np.roll(low, -1)

    # Identify bullish FVG patterns (High of i-1 < Low of i+1)
    bullish_mask = high_prev < low_next
    bullish_fvg_top[bullish_mask] = low_next[bullish_mask]
    bullish_fvg_bottom[bullish_mask] = high_prev[bullish_mask]

    # Identify bearish FVG patterns (Low of i-1 > High of i+1)
    bearish_mask = low_prev > high_next
    bearish_fvg_top[bearish_mask] = low_prev[bearish_mask]
    bearish_fvg_bottom[bearish_mask] = high_next[bearish_mask]

    # Set the first and last elements to NaN to avoid wraparound issues from np.roll
    for arr in [bullish_fvg_top, bullish_fvg_bottom, bearish_fvg_top, bearish_fvg_bottom]:
        arr[0] = arr[-1] = np.nan

    return bullish_fvg_top, bullish_fvg_bottom, bearish_fvg_top, bearish_fvg_bottom


class OneCandleExpansionStrategy(Strategy):
    """
    This strategy identifies a higher timeframe directional bias based on liquidity runs
    of previous key levels (e.g., Previous Day High/Low), then looks for a lower
    timeframe displacement (a strong move creating a Fair Value Gap), and enters on a
    retracement to that FVG.
    """
    # Strategy parameters
    min_rr = 3.0
    sl_buffer_pct = 0.01  # 1% buffer for stop loss placement

    def init(self):
        """
        Initialize the strategy's indicators and state variables.
        """
        # Pre-calculated daily data will be accessed via self.data attributes
        self.pdl = self.I(lambda x: x, self.data.PDL, name="PDL")
        self.pdh = self.I(lambda x: x, self.data.PDH, name="PDH")

        # FVG indicator
        self.bullish_fvg_top, self.bullish_fvg_bottom, \
        self.bearish_fvg_top, self.bearish_fvg_bottom = self.I(
            fvg_indicator, self.data.High, self.data.Low, name="FVG"
        )

        # State machine variables to track the setup
        self.liquidity_run_high = None
        self.liquidity_run_low = None
        self.displacement_fvg = None # Stores ('type', top, bottom)

    def next(self):
        """
        The main strategy logic that is executed on each bar.
        """
        # If a position is open, do nothing.
        if self.position:
            return

        # --- State Resets ---
        # If price moves away without the setup completing, reset the state.
        # For a bearish setup (after PDH run): if price makes a new high, invalidate.
        if self.liquidity_run_high and self.data.High[-1] > self.liquidity_run_high:
            self.liquidity_run_high = None
            self.displacement_fvg = None
        # For a bullish setup (after PDL run): if price makes a new low, invalidate.
        if self.liquidity_run_low and self.data.Low[-1] < self.liquidity_run_low:
            self.liquidity_run_low = None
            self.displacement_fvg = None

        # --- STAGE 1: Detect Liquidity Run ---
        # Look for a run on the Previous Day's High (for shorts)
        if self.data.High[-1] > self.pdh[-1] and not self.liquidity_run_high:
            self.liquidity_run_high = self.data.High[-1]
            self.liquidity_run_low = None  # Invalidate opposite signal
            self.displacement_fvg = None

        # Look for a run on the Previous Day's Low (for longs)
        elif self.data.Low[-1] < self.pdl[-1] and not self.liquidity_run_low:
            self.liquidity_run_low = self.data.Low[-1]
            self.liquidity_run_high = None  # Invalidate opposite signal
            self.displacement_fvg = None

        # --- STAGE 2: Detect Displacement FVG ---
        # After a PDH run, look for a *bearish* FVG as displacement
        if self.liquidity_run_high and not self.displacement_fvg:
            # Check the most recently formed FVG (on the candle that just closed)
            if not np.isnan(self.bearish_fvg_top[-2]):
                self.displacement_fvg = ('bearish', self.bearish_fvg_top[-2], self.bearish_fvg_bottom[-2])

        # After a PDL run, look for a *bullish* FVG as displacement
        elif self.liquidity_run_low and not self.displacement_fvg:
            if not np.isnan(self.bullish_fvg_top[-2]):
                self.displacement_fvg = ('bullish', self.bullish_fvg_top[-2], self.bullish_fvg_bottom[-2])

        # --- STAGE 3: Entry on Retracement ---
        if self.displacement_fvg:
            fvg_type, fvg_top, fvg_bottom = self.displacement_fvg

            # Bearish Entry: Retracement up into a bearish FVG
            if fvg_type == 'bearish' and self.data.High[-1] >= fvg_bottom:
                entry_price = self.data.Close[-1]
                sl = self.liquidity_run_high * (1 + self.sl_buffer_pct)
                tp = entry_price - (sl - entry_price) * self.min_rr

                # Ensure the trade has a valid R:R
                if tp < entry_price:
                    self.sell(sl=sl, tp=tp)
                    # Reset state after entry
                    self.liquidity_run_high = None
                    self.displacement_fvg = None

            # Bullish Entry: Retracement down into a bullish FVG
            elif fvg_type == 'bullish' and self.data.Low[-1] <= fvg_top:
                entry_price = self.data.Close[-1]
                sl = self.liquidity_run_low * (1 - self.sl_buffer_pct)
                tp = entry_price + (entry_price - sl) * self.min_rr

                if tp > entry_price:
                    self.buy(sl=sl, tp=tp)
                    # Reset state after entry
                    self.liquidity_run_low = None
                    self.displacement_fvg = None


if __name__ == '__main__':
    # --- Data Loading and Preprocessing ---
    data_path = 'data/BTC-USD-15m.csv'
    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.strip().title() for c in data.columns]
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # As a fallback, you might generate synthetic data or exit
        exit()

    # Calculate Previous Day High/Low (PDH/PDL)
    # Resample to daily timeframe to get the high and low of each day
    daily_data = data.resample('D').agg({
        'High': 'max',
        'Low': 'min'
    })

    # Shift the data to get the *previous* day's high and low
    daily_data['PDH'] = daily_data['High'].shift(1)
    daily_data['PDL'] = daily_data['Low'].shift(1)

    # Create a mapping from each date to the previous day's high and low
    pdl_map = daily_data['PDL'].to_dict()
    pdh_map = daily_data['PDH'].to_dict()

    # Map the values to the original dataframe's index
    # Using .normalize() removes the time component, leaving just the date
    data['PDL'] = data.index.normalize().map(pdl_map)
    data['PDH'] = data.index.normalize().map(pdh_map)

    # Forward-fill to propagate the daily values across the intraday bars
    # and back-fill to handle the first day's NaN
    data['PDL'].bfill(inplace=True)
    data['PDL'].ffill(inplace=True)
    data['PDH'].bfill(inplace=True)
    data['PDH'].ffill(inplace=True)

    # --- Backtesting ---
    bt = Backtest(data, OneCandleExpansionStrategy, cash=100_000, commission=.002)

    print("Running backtest...")
    stats = bt.run()
    print(stats)

    # --- Results Saving ---
    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON output
    # The stats object can contain non-serializable types (like pd.Timestamp)
    # and internal objects (_strategy, _equity_curve, etc.)
    sanitized_stats = {}
    for key, value in stats.items():
        if key.startswith('_'):
            continue
        try:
            # Convert numpy types to native Python types
            if isinstance(value, (np.int64, np.int32)):
                value = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                value = float(value)

            # Ensure the value is JSON serializable
            json.dumps(value)
            sanitized_stats[key] = value
        except (TypeError, OverflowError):
            sanitized_stats[key] = str(value) # fallback to string

    # Manually add strategy name
    sanitized_stats['strategy_name'] = 'one_candle_expansion'

    results_path = 'results/temp_result.json'
    with open(results_path, 'w') as f:
        json.dump(sanitized_stats, f, indent=2)

    print(f"\nBacktest stats saved to {results_path}")

    # --- Plotting ---
    plot_path = 'results/one_candle_expansion_plot.html'
    try:
        bt.plot(filename=plot_path)
        print(f"Backtest plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
