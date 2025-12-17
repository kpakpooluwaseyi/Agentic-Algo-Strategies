
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import json
import os

def sanitize_stats(stats):
    """
    Recursively sanitizes a dictionary or pandas Series containing stats,
    converting numpy types and handling non-serializable objects.
    """
    if isinstance(stats, (dict, pd.Series)):
        sanitized = {}
        for k, v in stats.items():
            if isinstance(v, (np.integer, np.int64)):
                sanitized[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                sanitized[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, (dict, pd.Series)):
                sanitized[k] = sanitize_stats(v)
            elif isinstance(v, (pd.DataFrame, pd.Index)):
                 # Skip DataFrames and Indexes as they are not easily serializable
                sanitized[k] = None
            elif isinstance(v, str):
                sanitized[k] = v
            else:
                try:
                    # Attempt to convert to a serializable type
                    json.dumps(v)
                    sanitized[k] = v
                except (TypeError, ValueError):
                    sanitized[k] = str(v) # Fallback to string representation
        return sanitized
    return stats

class SwingTradeConversionStrategy(Strategy):
    # --- Strategy Parameters ---
    m_pattern_lookback = 60  # Lookback period for M-pattern (number of bars)
    w_pattern_lookback = 60  # Lookback period for W-pattern (number of bars)
    sl_buffer_pct = 0.01  # Percentage buffer for stop-loss
    price_proximity_pct = 0.02 # Proximity for pivots in M/W patterns
    sl_to_be_pips = 50.0  # Pips of profit required to move SL to Break-Even
    enable_reversal_trade = False # Whether to immediately reverse the trade after a Level III exit

    def init(self):
        # --- State Variables ---
        self.trade_active = False
        self.sl_moved_to_be = False
        self.sl_moved_behind_asia = False
        self.mm_level = 0
        self.last_day_checked = None
        self.last_trade_was_long = None

    def next(self):
        # --- MM Level Counting & Exit Logic ---
        if self.position:
            self.last_trade_was_long = self.position.is_long
            today = self.data.index[-1].date()
            if self.last_day_checked is None:
                self.last_day_checked = today

            # Check for level progression once per day
            if today != self.last_day_checked:
                self.last_day_checked = today
                if self.position.is_long and self.data.High[-1] > self.data.Prev_Day_High[-1]:
                    self.mm_level += 1
                elif self.position.is_short and self.data.Low[-1] < self.data.Prev_Day_Low[-1]:
                    self.mm_level += 1

            # Exit at Level III
            if self.mm_level >= 3:
                self.position.close()
                return

        # --- Progressive Stop-Loss Management ---
        if self.position:
            trade = self.trades[0]
            current_hour = self.data.index[-1].hour

            # 3. Protect Profits at Level III
            if self.mm_level == 3:
                if self.position.is_long:
                    trade.sl = self.data.Low[-1] * (1 - self.sl_buffer_pct)
                else:
                    trade.sl = self.data.High[-1] * (1 + self.sl_buffer_pct)

            # 2. After London Open, move SL behind Asian Range
            if self.sl_moved_to_be and not self.sl_moved_behind_asia and current_hour >= 8:
                if self.position.is_long:
                    trade.sl = self.data.Asia_Low[-1] * (1 - self.sl_buffer_pct)
                else: # Short position
                    trade.sl = self.data.Asia_High[-1] * (1 + self.sl_buffer_pct)
                self.sl_moved_behind_asia = True

            # 1. Move SL to Break-Even
            if not self.sl_moved_to_be:
                pips = (self.data.Close[-1] - trade.entry_price) if self.position.is_long else (trade.entry_price - self.data.Close[-1])
                if pips > self.sl_to_be_pips:
                    trade.sl = trade.entry_price
                    self.sl_moved_to_be = True

        # --- Entry Logic ---
        if not self.position and not self.trade_active:
            # --- W-Pattern (Long Entry) ---
            w_window = self.data.Low[-self.w_pattern_lookback:]
            peaks, _ = find_peaks(-w_window, distance=5)

            if len(peaks) >= 2:
                trough1_idx, trough2_idx = peaks[-2], peaks[-1]
                trough1_price, trough2_price = w_window[trough1_idx], w_window[trough2_idx]

                if trough2_idx > trough1_idx + 1:
                    center_peak_window = self.data.High[-self.w_pattern_lookback+trough1_idx+1:-self.w_pattern_lookback+trough2_idx]
                    if len(center_peak_window) > 0:
                        center_peak_price = np.max(center_peak_window)

                        if abs(trough1_price - trough2_price) / trough1_price < self.price_proximity_pct and self.data.Close[-1] > center_peak_price:
                            sl = min(trough1_price, trough2_price) * (1 - self.sl_buffer_pct)
                            self.buy(sl=sl)
                            self.trade_active = True
                            self.sl_moved_to_be = False
                            self.sl_moved_behind_asia = False
                            self.mm_level = 0
                            self.last_day_checked = None
                            return

            # --- M-Pattern (Short Entry) ---
            m_window = self.data.High[-self.m_pattern_lookback:]
            peaks, _ = find_peaks(m_window, distance=5)

            if len(peaks) >= 2:
                peak1_idx, peak2_idx = peaks[-2], peaks[-1]
                peak1_price, peak2_price = m_window[peak1_idx], m_window[peak2_idx]

                if peak2_idx > peak1_idx + 1:
                    center_trough_window = self.data.Low[-self.m_pattern_lookback+peak1_idx+1:-self.m_pattern_lookback+peak2_idx]
                    if len(center_trough_window) > 0:
                        center_trough_price = np.min(center_trough_window)

                        if abs(peak1_price - peak2_price) / peak1_price < self.price_proximity_pct and self.data.Close[-1] < center_trough_price:
                            sl = max(peak1_price, peak2_price) * (1 + self.sl_buffer_pct)
                            self.sell(sl=sl)
                            self.trade_active = True
                            self.sl_moved_to_be = False
                            self.sl_moved_behind_asia = False
                            self.mm_level = 0
                            self.last_day_checked = None
                            return

        # --- Reset flags when position is closed ---
        if not self.position and self.trade_active:
            # Check if the trade was closed by the Level III exit
            if self.enable_reversal_trade and self.mm_level >= 3 and self.last_trade_was_long is not None:
                # Immediate reversal trade with a defined SL
                if self.last_trade_was_long: # Previous trade was long, so reverse to short
                    sl = self.data.High[-1] * (1 + self.sl_buffer_pct)
                    self.sell(sl=sl)
                else: # Previous trade was short, so reverse to long
                    sl = self.data.Low[-1] * (1 - self.sl_buffer_pct)
                    self.buy(sl=sl)

                # A new trade was initiated, so we reset flags for it but don't deactivate trading.
                self.sl_moved_to_be = False
                self.sl_moved_behind_asia = False
                self.mm_level = 0
                self.last_day_checked = None
                return # Exit to avoid the self.trade_active = False below

            self.trade_active = False
            self.sl_moved_to_be = False
            self.sl_moved_behind_asia = False
            self.mm_level = 0
            self.last_day_checked = None

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Clean column names: strip whitespace and capitalize
    data.columns = [col.strip().title() for col in data.columns]

    # Drop the junk 'Unnamed: 6' column if it exists
    if 'Unnamed: 6' in data.columns:
        data.drop(columns=['Unnamed: 6'], inplace=True)

    # Pre-process data to add session information
    data['Hour'] = data.index.hour
    data['Is_Asian_Session'] = (data['Hour'] >= 0) & (data['Hour'] < 8)

    # Calculate daily Asian session stats using a robust merge strategy
    asia_stats = data[data['Is_Asian_Session']].groupby(data[data['Is_Asian_Session']].index.date).agg(
        Daily_Asia_High=('High', 'max'),
        Daily_Asia_Low=('Low', 'min')
    )

    # Create a temporary date column for merging all daily stats
    data['merge_date'] = data.index.date

    # Merge both sets of daily stats at once
    data = pd.merge(data, asia_stats, left_on='merge_date', right_index=True, how='left')

    daily_stats = data.groupby('merge_date').agg(
        Daily_High=('High', 'max'),
        Daily_Low=('Low', 'min')
    )
    daily_stats['Prev_Day_High'] = daily_stats['Daily_High'].shift(1)
    daily_stats['Prev_Day_Low'] = daily_stats['Daily_Low'].shift(1)
    data = pd.merge(data, daily_stats[['Prev_Day_High', 'Prev_Day_Low']], left_on='merge_date', right_index=True, how='left')

    # Clean up and forward-fill
    data.drop(columns=['merge_date'], inplace=True)
    data.rename(columns={'Daily_Asia_High': 'Asia_High', 'Daily_Asia_Low': 'Asia_Low'}, inplace=True)
    data['Asia_High'] = data['Asia_High'].ffill()
    data['Asia_Low'] = data['Asia_Low'].ffill()
    data['Prev_Day_High'] = data['Prev_Day_High'].ffill()
    data['Prev_Day_Low'] = data['Prev_Day_Low'].ffill()

    # Drop rows with NaN values that couldn't be forward-filled (e.g., the first day)
    data.dropna(inplace=True)

    print("Data pre-processing complete. DataFrame head:")
    print(data.head())

    bt = Backtest(data, SwingTradeConversionStrategy, cash=100_000, commission=.002, finalize_trades=True)

    # --- Optimization ---
    print("Optimizing strategy...")
    stats = bt.optimize(
        m_pattern_lookback=range(20, 100, 10),
        w_pattern_lookback=range(20, 100, 10),
        price_proximity_pct=[i/100 for i in range(1, 5)], # 1% to 4%
        maximize='Sharpe Ratio',
        constraint=lambda p: p.m_pattern_lookback >= 20 and p.w_pattern_lookback >= 20
    )

    print("Best optimization stats:")
    print(stats)

    # Sanitize and save the results
    sanitized_result = sanitize_stats(stats)

    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_result, f, indent=4)

    print("\\nResults saved to results/temp_result.json")

    # Generate and save the plot
    plot_filename = 'results/swing_trade_conversion.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")
