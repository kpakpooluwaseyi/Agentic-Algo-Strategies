import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks
import json
import os

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

    # Bullish FVG: High of i-1 < Low of i+1
    bullish_mask = high_prev < low_next
    bullish_fvg_top[bullish_mask] = low_next[bullish_mask]
    bullish_fvg_bottom[bullish_mask] = high_prev[bullish_mask]

    # Bearish FVG: Low of i-1 > High of i+1
    bearish_mask = low_prev > high_next
    bearish_fvg_top[bearish_mask] = low_prev[bearish_mask]
    bearish_fvg_bottom[bearish_mask] = high_next[bearish_mask]

    for arr in [bullish_fvg_top, bullish_fvg_bottom, bearish_fvg_top, bearish_fvg_bottom]:
        arr[0] = arr[-1] = np.nan

    return bullish_fvg_top, bullish_fvg_bottom, bearish_fvg_top, bearish_fvg_bottom

# --- Data Pre-processing ---

def preprocess_data(df: pd.DataFrame):
    """
    Pre-processes the 15m data to include daily (HTF) context.
    """
    print("Columns inside preprocess_data:", df.columns)
    # 1. Resample to Daily to get HTF context
    daily_df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    # 2. Calculate Daily Swing Points
    daily_highs, daily_lows = swing_indicator(daily_df['Close'].to_numpy(), distance=5)
    daily_df['is_daily_swing_high'] = daily_highs
    daily_df['is_daily_swing_low'] = daily_lows

    # 3. Calculate Daily FVGs
    d_bfvg_t, d_bfvg_b, d_bearfvg_t, d_bearfvg_b = fvg_indicator(daily_df['High'].to_numpy(), daily_df['Low'].to_numpy())
    daily_df['daily_bullish_fvg_top'] = d_bfvg_t
    daily_df['daily_bullish_fvg_bottom'] = d_bfvg_b
    daily_df['daily_bearish_fvg_top'] = d_bearfvg_t
    daily_df['daily_bearish_fvg_bottom'] = d_bearfvg_b

    # 4. Map daily data back to the 15m dataframe
    df['date'] = df.index.date
    daily_df['date'] = daily_df.index.date

    # Select only the columns to merge
    daily_context_cols = [
        'date', 'is_daily_swing_high', 'is_daily_swing_low',
        'daily_bullish_fvg_top', 'daily_bullish_fvg_bottom',
        'daily_bearish_fvg_top', 'daily_bearish_fvg_bottom'
    ]
    merged_df = pd.merge(df, daily_df[daily_context_cols], on='date', how='left')
    merged_df.index = df.index

    # Forward-fill the daily data to apply it to each 15m candle
    merged_df[daily_context_cols[1:]] = merged_df[daily_context_cols[1:]].ffill()

    return merged_df.drop(columns=['date'])


# --- Strategy Class ---

class AplusSetupStrategy(Strategy):
    # Strategy parameters
    rr_ratio = 3.0
    ltf_swing_distance = 15
    trade_size = 0.05

    def init(self):
        # LTF (15m) Indicators
        self.ltf_swing_highs, self.ltf_swing_lows = self.I(
            swing_indicator, self.data.Close, self.ltf_swing_distance
        )
        self.bullish_fvg_top, self.bullish_fvg_bottom, \
        self.bearish_fvg_top, self.bearish_fvg_bottom = self.I(
            fvg_indicator, self.data.High, self.data.Low, name="M15_FVG"
        )

        # State machine variables
        self.htf_poi_touched = None # ('bullish' or 'bearish', price_level)
        self.liquidity_sweep_price = None
        self.displacement_fvg = None # ('bullish' or 'bearish', top, bottom)

    def next(self):
        # Reset state if we are no longer in a position (i.e., a trade just closed)
        if not self.position and self.liquidity_sweep_price is not None:
            self.reset_state()

        # Only check for entries if we are not in a position
        if self.position:
            return

        # STATE 1: Wait for price to interact with an HTF Point of Interest (POI)
        if self.htf_poi_touched is None:
            # Check for bullish POI interaction (Discount zones)
            # 1. Price drops into a Daily Bullish FVG
            if not np.isnan(self.data.daily_bullish_fvg_top[-1]) and \
               self.data.Low[-1] <= self.data.daily_bullish_fvg_top[-1]:
                self.htf_poi_touched = ('bullish', self.data.daily_bullish_fvg_bottom[-1])
                return

            # 2. Price retests a recent Daily Swing Low
            # We check for a swing low confirmed on the *previous* daily candle to avoid lookahead
            if self.data.is_daily_swing_low[-2]:
                 # Find the price of that swing low. This is a simplification.
                 # A more complex implementation would find the exact low of that day.
                 daily_low_price = self.data.Low.s.resample('D').min().shift(1).ffill().iloc[-1]
                 if abs(self.data.Low[-1] - daily_low_price) / daily_low_price < 0.005: # within 0.5%
                    self.htf_poi_touched = ('bullish', daily_low_price)
                    return

            # Check for bearish POI interaction (Premium zones)
            # 1. Price rises into a Daily Bearish FVG
            if not np.isnan(self.data.daily_bearish_fvg_bottom[-1]) and \
               self.data.High[-1] >= self.data.daily_bearish_fvg_bottom[-1]:
                self.htf_poi_touched = ('bearish', self.data.daily_bearish_fvg_top[-1])
                return

            # 2. Price retests a recent Daily Swing High
            if self.data.is_daily_swing_high[-2]:
                daily_high_price = self.data.High.s.resample('D').max().shift(1).ffill().iloc[-1]
                if abs(self.data.High[-1] - daily_high_price) / daily_high_price < 0.005: # within 0.5%
                    self.htf_poi_touched = ('bearish', daily_high_price)
                    return
        # STATE 2: Wait for a liquidity sweep of a recent LTF swing point
        if self.htf_poi_touched is not None and self.liquidity_sweep_price is None:
            poi_type, poi_price = self.htf_poi_touched

            if poi_type == 'bullish':
                # Find the most recent LTF (M15) swing low to be swept
                recent_lows = np.where(self.ltf_swing_lows[:-1])[0]
                if not recent_lows.any(): return

                last_swing_low_idx = recent_lows[-1]
                last_swing_low_price = self.data.Low[last_swing_low_idx]

                # Check if the current bar sweeps that low
                if self.data.Low[-1] < last_swing_low_price:
                    self.liquidity_sweep_price = self.data.Low[-1] # Mark the sweep

            elif poi_type == 'bearish':
                # Find the most recent LTF (M15) swing high to be swept
                recent_highs = np.where(self.ltf_swing_highs[:-1])[0]
                if not recent_highs.any(): return

                last_swing_high_idx = recent_highs[-1]
                last_swing_high_price = self.data.High[last_swing_high_idx]

                if self.data.High[-1] > last_swing_high_price:
                    self.liquidity_sweep_price = self.data.High[-1]

        # STATE 3: Look for displacement (a new FVG) and place entry order
        if self.liquidity_sweep_price is not None and self.displacement_fvg is None:
            poi_type, _ = self.htf_poi_touched

            # After a bullish setup (sweep of a low), we look for a bullish FVG
            if poi_type == 'bullish' and not np.isnan(self.bullish_fvg_top[-2]):
                self.displacement_fvg = ('bullish', self.bullish_fvg_top[-2], self.bullish_fvg_bottom[-2])

            # After a bearish setup (sweep of a high), we look for a bearish FVG
            elif poi_type == 'bearish' and not np.isnan(self.bearish_fvg_top[-2]):
                self.displacement_fvg = ('bearish', self.bearish_fvg_top[-2], self.bearish_fvg_bottom[-2])

        # STATE 4: Execute entry if price retraces to the displacement FVG
        if self.displacement_fvg is not None:
            fvg_type, fvg_top, fvg_bottom = self.displacement_fvg

            if fvg_type == 'bullish':
                entry_price = fvg_top
                sl = self.liquidity_sweep_price # SL below the liquidity sweep low
                tp = entry_price + (entry_price - sl) * self.rr_ratio

                if self.data.Low[-1] <= entry_price:
                    if entry_price > sl:
                         self.buy(sl=sl, tp=tp, limit=entry_price, size=self.trade_size)

            elif fvg_type == 'bearish':
                entry_price = fvg_bottom
                sl = self.liquidity_sweep_price # SL above the liquidity sweep high
                tp = entry_price - (sl - entry_price) * self.rr_ratio

                if self.data.High[-1] >= entry_price:
                    if entry_price < sl:
                        self.sell(sl=sl, tp=tp, limit=entry_price, size=self.trade_size)

    def reset_state(self):
        """Helper function to reset the state machine variables."""
        self.htf_poi_touched = None
        self.liquidity_sweep_price = None
        self.displacement_fvg = None

# --- Backtesting Execution ---

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        # Load data
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.strip().title() for c in data.columns]
        if 'Unnamed: 6' in data.columns:
            data = data.drop(columns=['Unnamed: 6'])

        # Pre-process data to add HTF context
        data = preprocess_data(data)
        data.dropna(inplace=True) # Drop rows where HTF context is not available

        # Run backtest
        bt = Backtest(data, AplusSetupStrategy, cash=100_000, commission=.002)

        print("Running backtest...")
        stats = bt.run()
        print(stats)

        # Save results
        os.makedirs('results', exist_ok=True)

        # Sanitize stats for JSON output
        sanitized_stats = {key: (val if isinstance(val, (int, float, str, bool)) or val is None else str(val))
                           for key, val in stats.items() if not key.startswith('_')}

        result_summary = {
            'strategy_name': 'a_plus_setup_four_step_framework',
            'return': sanitized_stats.get('Return [%]'),
            'sharpe': sanitized_stats.get('Sharpe Ratio'),
            'max_drawdown': sanitized_stats.get('Max. Drawdown [%]'),
            'win_rate': sanitized_stats.get('Win Rate [%]'),
            'total_trades': sanitized_stats.get('# Trades')
        }

        with open('results/temp_result.json', 'w') as f:
            json.dump(result_summary, f, indent=4)

        # Generate plot
        try:
            bt.plot(filename='results/a_plus_setup_four_step_framework.html')
            print("Plot saved to results/a_plus_setup_four_step_framework.html")
        except Exception as e:
            print(f"Could not generate plot: {e}")
