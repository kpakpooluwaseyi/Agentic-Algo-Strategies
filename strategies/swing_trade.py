from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import json

def preprocess_data(df, peak_distance=20, peak_prominence=0.02):
    """
    Calculates Asian session data and swing points for the strategy.
    """
    df_copy = df.copy()

    # 1. Asian Session Calculation
    df_copy['hour'] = df_copy.index.hour
    df_copy['date'] = df_copy.index.date
    is_asia = (df_copy['hour'] >= 0) & (df_copy['hour'] < 8)
    asia_session_data = df_copy[is_asia].groupby('date').agg(
        asia_high=('High', 'max'),
        asia_low=('Low', 'min')
    ).shift(1)
    df_copy['prev_asia_high'] = df_copy['date'].map(asia_session_data['asia_high'])
    df_copy['prev_asia_low'] = df_copy['date'].map(asia_session_data['asia_low'])
    df_copy['prev_asia_high'] = df_copy['prev_asia_high'].ffill()
    df_copy['prev_asia_low'] = df_copy['prev_asia_low'].ffill()

    # 2. Swing Point Calculation
    close_series = df_copy['Close']
    price_range = close_series.max() - close_series.min()
    actual_prominence = price_range * peak_prominence

    high_peaks_indices, _ = find_peaks(close_series, distance=peak_distance, prominence=actual_prominence)
    low_peaks_indices, _ = find_peaks(-close_series, distance=peak_distance, prominence=actual_prominence)

    df_copy['swing_high'] = np.nan
    df_copy.iloc[high_peaks_indices, df_copy.columns.get_loc('swing_high')] = close_series.iloc[high_peaks_indices]

    df_copy['swing_low'] = np.nan
    df_copy.iloc[low_peaks_indices, df_copy.columns.get_loc('swing_low')] = close_series.iloc[low_peaks_indices]

    # Clean up helper columns and drop rows where lookback data is unavailable
    df_copy = df_copy.drop(columns=['hour', 'date'])
    df_copy = df_copy.dropna(subset=['prev_asia_high', 'prev_asia_low'])
    return df_copy

def passthrough(series, *args, **kwargs):
    return series

class SwingTradeStrategy(Strategy):
    # Optimizable parameters for M/W pattern detection
    peak_distance = 20
    peak_prominence = 0.02 # As a percentage of the price range

    # Optimizable parameters for risk management (percentage-based)
    profit_pct_to_be = 0.5   # 0.5% profit to move SL to break-even
    level_1_bars = 4 * 8     # 8 hours
    london_hunt_bars = 4 * 3  # 3 hours
    reversal_exit_pct = 1.5  # 1.5% reversal from peak to exit in Level III

    # Parameters for 3-day cycle proxy
    cycle_lookback_bars = 96 * 3 # 3 days of 15-min candles
    entry_window_bars = 96     # Allow entries for 1 day after peak formation

    def init(self):
        # Make pre-processed data available to the strategy
        self.prev_asia_high = self.I(passthrough, self.data.df['prev_asia_high'])
        self.prev_asia_low = self.I(passthrough, self.data.df['prev_asia_low'])
        self.swing_highs = self.I(passthrough, self.data.df['swing_high'])
        self.swing_lows = self.I(passthrough, self.data.df['swing_low'])

        # State machine variables for patterns
        self.recent_swings = []
        self.m_pattern = None
        self.w_pattern = None

        # State machine variables for trade management
        self.trade_level = 0
        self.bars_in_trade = 0
        self.trade_peak_price = 0

        # State for 3-day cycle
        self.peak_formation_bar = 0
        self.peak_type = None

    def next(self):
        current_bar = len(self.data.Close) - 1

        # === Trade Management Logic ===
        if self.position:
            self.bars_in_trade += 1
            trade = self.trades[0]

            # Level 0 -> 1: Move SL to Break-Even
            if self.trade_level == 0:
                profit_pct = (self.data.Close[-1] / trade.entry_price - 1) * 100 if self.position.is_long else (1 - self.data.Close[-1] / trade.entry_price) * 100
                if profit_pct >= self.profit_pct_to_be:
                    trade.sl = trade.entry_price
                    self.trade_level = 1

            # Level 1 & 2: Time-based SL adjustments
            if self.trade_level == 1 and self.bars_in_trade > self.level_1_bars: self.trade_level = 2
            if self.trade_level == 2 and self.bars_in_trade > (self.level_1_bars + self.london_hunt_bars):
                if self.position.is_long: trade.sl = self.prev_asia_low[-1]
                else: trade.sl = self.prev_asia_high[-1]
                self.trade_level = 3
                self.trade_peak_price = self.data.High[-1] if self.position.is_long else self.data.Low[-1]

            # Level 3: "Stop Hunt" Reversal Exit
            if self.trade_level == 3:
                if self.position.is_long:
                    self.trade_peak_price = max(self.trade_peak_price, self.data.High[-1])
                    reversal_threshold = self.trade_peak_price * (1 - self.reversal_exit_pct / 100)
                    if self.data.Close[-1] < reversal_threshold:
                        self.position.close()
                else: # Short position
                    self.trade_peak_price = min(self.trade_peak_price, self.data.Low[-1])
                    reversal_threshold = self.trade_peak_price * (1 + self.reversal_exit_pct / 100)
                    if self.data.Close[-1] > reversal_threshold:
                        self.position.close()
            return

        # Reset trade management state when position is closed
        self.trade_level = 0
        self.bars_in_trade = 0
        self.trade_peak_price = 0

        # === 3-Day Cycle Peak Formation Logic ===
        if current_bar > self.cycle_lookback_bars:
            lookback_high = self.data.High[-self.cycle_lookback_bars:].max()
            lookback_low = self.data.Low[-self.cycle_lookback_bars:].min()

            # A new peak high is formed if the current high is the highest in the lookback period
            if self.data.High[-1] == lookback_high:
                self.peak_formation_bar = current_bar
                self.peak_type = 'high'

            # A new peak low is formed if the current low is the lowest
            elif self.data.Low[-1] == lookback_low:
                self.peak_formation_bar = current_bar
                self.peak_type = 'low'

        # === Pattern Detection Logic ===
        if not np.isnan(self.swing_highs[-1]):
            new_swing = ('high', self.swing_highs[-1], current_bar)
            if not self.recent_swings or self.recent_swings[-1][0] != 'high': self.recent_swings.append(new_swing)
        if not np.isnan(self.swing_lows[-1]):
            new_swing = ('low', self.swing_lows[-1], current_bar)
            if not self.recent_swings or self.recent_swings[-1][0] != 'low': self.recent_swings.append(new_swing)
        if len(self.recent_swings) > 5: self.recent_swings.pop(0)

        # M-Pattern
        if len(self.recent_swings) >= 3 and all(s[0] == t for s, t in zip(self.recent_swings[-3:], ['high', 'low', 'high'])):
            p1, v, p2 = self.recent_swings[-3][1], self.recent_swings[-2][1], self.recent_swings[-1][1]
            if p2 < p1: self.m_pattern, self.w_pattern = {'trigger_price': v, 'sl': p1, 'active': True}, None
        # W-Pattern
        if len(self.recent_swings) >= 3 and all(s[0] == t for s, t in zip(self.recent_swings[-3:], ['low', 'high', 'low'])):
            v1, p, v2 = self.recent_swings[-3][1], self.recent_swings[-2][1], self.recent_swings[-1][1]
            if v2 > v1: self.w_pattern, self.m_pattern = {'trigger_price': p, 'sl': v1, 'active': True}, None

        # === Entry Logic with 3-Day Cycle Filter ===
        is_in_entry_window = (current_bar - self.peak_formation_bar) <= self.entry_window_bars

        # Sell on M-Pattern only after a Peak Formation High
        if self.m_pattern and self.m_pattern['active'] and self.peak_type == 'high' and is_in_entry_window:
            if self.data.Close[-1] < self.m_pattern['trigger_price']:
                self.sell(sl=self.m_pattern['sl'])
                self.m_pattern['active'] = False

        # Buy on W-Pattern only after a Peak Formation Low
        if self.w_pattern and self.w_pattern['active'] and self.peak_type == 'low' and is_in_entry_window:
            if self.data.Close[-1] > self.w_pattern['trigger_price']:
                self.buy(sl=self.w_pattern['sl'])
                self.w_pattern['active'] = False


if __name__ == '__main__':
    data = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)

    # Robustly clean and standardize column names
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data.columns = [col.strip().title() for col in data.columns]

    # Preprocess the data to include Asian session info and swing points
    data = preprocess_data(data, peak_distance=20, peak_prominence=0.02)

    bt = Backtest(data, SwingTradeStrategy, cash=100_000, commission=.002)
    stats = bt.run()

    print(stats)

    import os
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
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
            elif isinstance(value, pd.Timestamp):
                sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                sanitized[key] = str(value)
            else:
                sanitized[key] = value
        # Remove non-serializable objects
        if '_strategy' in sanitized:
            del sanitized['_strategy']
        if '_equity_curve' in sanitized:
            del sanitized['_equity_curve']
        if '_trades' in sanitized:
            del sanitized['_trades']
        return sanitized

    clean_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(clean_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    try:
        bt.plot(filename='results/swing_trade.html')
    except Exception as e:
        print(f"Could not generate plot: {e}")
