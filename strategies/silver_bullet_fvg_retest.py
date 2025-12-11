from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks


def swing_indicator(price: np.ndarray, distance: int):
    """
    Identifies swing highs and lows using scipy.signal.find_peaks.
    Returns boolean arrays marking the locations of swing points.
    """
    # Find peaks (swing highs) and troughs (swing lows)
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

    Returns four arrays:
    - bullish_fvg_top, bullish_fvg_bottom
    - bearish_fvg_top, bearish_fvg_bottom
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


class SilverBulletFvgRetestStrategy(Strategy):
    # Parameters
    stop_loss_pct = 0.5
    swing_distance = 10
    fvg_invalidation_pct = 1.0

    def init(self):
        # Initialize indicators using self.I()
        self.bullish_fvg_top, self.bullish_fvg_bottom, \
        self.bearish_fvg_top, self.bearish_fvg_bottom = self.I(
            fvg_indicator, self.data.High, self.data.Low, name="FVG"
        )

        self.swing_highs, self.swing_lows = self.I(
            swing_indicator, self.data.Close, self.swing_distance
        )

        # To track the most recent FVG and trade state
        self.active_fvg = None
        self.entry_price = None
        self.final_tp = None

    def next(self):
        # 1. EXIT LOGIC: Manage open positions first
        if self.position:
            # For long positions, check TP targets
            if self.position.is_long and self.data.High[-1] >= self.final_tp:
                self.position.close()
            # For short positions, check TP targets
            elif self.position.is_short and self.data.Low[-1] <= self.final_tp:
                self.position.close()
            return # Don't look for new entries if a position is open

        # Reset state if no position
        self.active_fvg = None
        self.entry_price = None
        self.final_tp = None

        # 2. TIME-BASED FILTER
        current_time = self.data.index[-1].time()
        if not (pd.Timestamp("10:00").time() <= current_time < pd.Timestamp("11:00").time()):
            return

        # 3. FVG DETECTION
        if not np.isnan(self.bullish_fvg_top[-2]):
            self.active_fvg = ('bullish', self.bullish_fvg_top[-2], self.bullish_fvg_bottom[-2])
        elif not np.isnan(self.bearish_fvg_top[-2]):
            self.active_fvg = ('bearish', self.bearish_fvg_top[-2], self.bearish_fvg_bottom[-2])

        # Invalidate the FVG if the price has moved too far away
        if self.active_fvg:
            fvg_type, fvg_top, fvg_bottom = self.active_fvg
            if (fvg_type == 'bullish' and self.data.Low[-1] > fvg_top + self.fvg_invalidation_pct) or \
               (fvg_type == 'bearish' and self.data.High[-1] < fvg_bottom - self.fvg_invalidation_pct):
                self.active_fvg = None

        # 4. ENTRY LOGIC
        if self.active_fvg and not self.position:
            fvg_type, fvg_top, fvg_bottom = self.active_fvg

            if fvg_type == 'bullish' and self.data.Low[-1] <= fvg_top:
                self.entry_price = self.data.Close[-1]
                sl = fvg_bottom * (1 - self.stop_loss_pct / 100)
                primary_tp = self.entry_price + 5
                secondary_tp = primary_tp # Fallback

                # Find the nearest swing high *above* the entry price
                past_swing_highs_indices = np.where(self.swing_highs[:len(self.data.Close)-1])[0]
                if past_swing_highs_indices.any():
                    valid_targets = self.data.High[past_swing_highs_indices][self.data.High[past_swing_highs_indices] > self.entry_price]
                    if valid_targets.any():
                        secondary_tp = np.min(valid_targets)

                # Final TP is the max of the two, ensuring 5-handle minimum
                self.final_tp = max(primary_tp, secondary_tp)

                if self.entry_price > sl:
                    self.buy(sl=sl)

            elif fvg_type == 'bearish' and self.data.High[-1] >= fvg_bottom:
                self.entry_price = self.data.Close[-1]
                sl = fvg_top * (1 + self.stop_loss_pct / 100)
                primary_tp = self.entry_price - 5
                secondary_tp = primary_tp # Fallback

                # Find the nearest swing low *below* the entry price
                past_swing_lows_indices = np.where(self.swing_lows[:len(self.data.Close)-1])[0]
                if past_swing_lows_indices.any():
                    valid_targets = self.data.Low[past_swing_lows_indices][self.data.Low[past_swing_lows_indices] < self.entry_price]
                    if valid_targets.any():
                        secondary_tp = np.max(valid_targets)

                # Final TP is the min of the two, ensuring 5-handle minimum
                self.final_tp = min(primary_tp, secondary_tp)

                if self.entry_price < sl:
                    self.sell(sl=sl)

def generate_synthetic_data(days=50):
    """
    Generates synthetic 5-minute data with a guaranteed bullish FVG and retracement
    during the 10:00-11:00 AM NY time window on specific days.
    """
    n = days * 24 * 12  # 5-minute intervals in a day
    start_date = "2023-01-01"
    index = pd.date_range(start=start_date, periods=n, freq='5min', tz='America/New_York')

    # Base random data
    base_price = 100
    returns = np.random.normal(loc=0, scale=0.001, size=n)
    price = base_price * (1 + returns).cumprod()

    df = pd.DataFrame(index=index)
    df['Open'] = price
    df['High'] = df['Open'] + np.random.uniform(0, 0.05, n)
    df['Low'] = df['Open'] - np.random.uniform(0, 0.05, n)
    df['Close'] = df['Open'] + np.random.normal(loc=0, scale=0.0005, size=n)

    # Inject a clear FVG pattern on a specific day at 10:00 AM
    # We will inject this pattern every 5 days to have multiple occurrences
    for i in range(0, days, 5):
        target_day = pd.Timestamp(start_date, tz='America/New_York') + pd.Timedelta(days=i)
        start_time = target_day.replace(hour=10, minute=5)

        try:
            start_idx = df.index.get_loc(start_time)
        except KeyError:
            continue # Skip if the exact time is not in the index

        # Candle 1 (i-1): The candle before the big move
        # Let's use existing data

        # Candle 2 (i): The displacement candle
        df.loc[df.index[start_idx], 'Open'] = df.loc[df.index[start_idx-1], 'Close']
        df.loc[df.index[start_idx], 'High'] = df.loc[df.index[start_idx], 'Open'] + 2.0
        df.loc[df.index[start_idx], 'Low'] = df.loc[df.index[start_idx], 'Open']
        df.loc[df.index[start_idx], 'Close'] = df.loc[df.index[start_idx], 'High'] - 0.1

        # Candle 3 (i+1): The candle after, creating the gap
        # The low of this candle is higher than the high of candle 1
        df.loc[df.index[start_idx+1], 'Open'] = df.loc[df.index[start_idx], 'Close']
        df.loc[df.index[start_idx+1], 'Low'] = df.loc[df.index[start_idx-1], 'High'] + 0.2
        df.loc[df.index[start_idx+1], 'High'] = df.loc[df.index[start_idx+1], 'Low'] + 0.5
        df.loc[df.index[start_idx+1], 'Close'] = df.loc[df.index[start_idx+1], 'High'] - 0.1

        # Retracement candle (i+2) that trades into the FVG
        fvg_bottom = df.loc[df.index[start_idx-1], 'High']
        df.loc[df.index[start_idx+2], 'Open'] = df.loc[df.index[start_idx+1], 'Close']
        df.loc[df.index[start_idx+2], 'High'] = df.loc[df.index[start_idx+2], 'Open'] + 0.1
        df.loc[df.index[start_idx+2], 'Low'] = fvg_bottom + 0.05 # Trade into the FVG
        df.loc[df.index[start_idx+2], 'Close'] = df.loc[df.index[start_idx+2], 'Low'] + 0.1

    return df


if __name__ == '__main__':
    # Load or generate data
    data = generate_synthetic_data()

    # Run backtest
    bt = Backtest(data, SilverBulletFvgRetestStrategy, cash=10000, commission=.002)

    # Optimize
    sl_range = list(np.arange(0.1, 1.0, 0.1))
    fvg_range = list(np.arange(0.5, 2.0, 0.5))
    stats = bt.optimize(stop_loss_pct=sl_range,
                        swing_distance=range(5, 20, 5),
                        fvg_invalidation_pct=fvg_range,
                        maximize='Sharpe Ratio',
                        max_tries=300)

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Manually extract the required fields
    results = {
        'strategy_name': 'silver_bullet_fvg_retest',
        'return': float(stats.get('Return [%]', 0.0)),
        'sharpe': float(stats.get('Sharpe Ratio', 0.0)),
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
        'win_rate': float(stats.get('Win Rate [%]', 0.0)),
        'total_trades': int(stats.get('# Trades', 0))
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate plot
    try:
        bt.plot(filename='results/silver_bullet_plot.html')
    except Exception as e:
        print(f"Could not generate plot: {e}")
