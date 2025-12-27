
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def preprocess_data(df: pd.DataFrame, peak_distance=10) -> pd.DataFrame:
    """
    Adds session information and identifies swing points for M/W pattern analysis.
    """
    # Using UTC time for session definitions, as is common with 24/7 crypto data.
    # "Asia Range" proxy: 00:00 - 08:00 UTC
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    df['is_asia'] = (df['hour'] >= 0) & (df['hour'] < 8)

    # Calculate previous day's Asia session high/low
    asia_session_data = df[df['is_asia']].groupby('date').agg(
        prev_asia_high=('High', 'max'),
        prev_asia_low=('Low', 'min')
    ).shift(1) # Shift to avoid lookahead bias

    df = pd.merge(df, asia_session_data, on='date', how='left')
    df['prev_asia_high'] = df['prev_asia_high'].ffill()
    df['prev_asia_low'] = df['prev_asia_low'].ffill()

    # Identify swing points for M/W pattern recognition
    high_peaks_indices, _ = find_peaks(df['High'], distance=peak_distance)
    low_peaks_indices, _ = find_peaks(-df['Low'], distance=peak_distance)

    df['is_swing_high'] = False
    df.iloc[high_peaks_indices, df.columns.get_loc('is_swing_high')] = True
    df['is_swing_low'] = False
    df.iloc[low_peaks_indices, df.columns.get_loc('is_swing_low')] = True

    # Add a simple check for London session proxy
    df['is_london'] = (df['hour'] >= 8) & (df['hour'] < 16)


    df = df.drop(columns=['date', 'is_asia']).dropna() # Keep 'hour' for time-based exits
    return df

def passthrough(series):
    return series

from enum import Enum

class State(Enum):
    """
    Defines the possible states of the strategy's pattern detection state machine.
    """
    WAITING_FOR_SWIPE = 1
    M_PEAK_ONE = 2
    M_CENTER_TROUGH = 3
    M_PEAK_TWO = 4
    W_TROUGH_ONE = 5
    W_CENTER_PEAK = 6
    W_TROUGH_TWO = 7
    CHECK_CONSOLIDATION_M = 8
    CHECK_CONSOLIDATION_W = 9

class MarketMakerCycleReversalStrategy(Strategy):
    # Optimizable parameters
    m_pattern_max_bars = 50
    w_pattern_max_bars = 50
    consolidation_bars = 4  # 60 minutes for 15m timeframe
    consolidation_range_pct = 0.005 # 0.5% range
    risk_reward_ratio = 2.0
    sl_buffer_pct = 0.001 # 0.1% buffer for SL

    def init(self):
        self.is_london = self.I(passthrough, self.data.df['is_london'].values, name="is_london")
        self.prev_asia_high = self.I(passthrough, self.data.df['prev_asia_high'].values, name="prev_asia_high")
        self.prev_asia_low = self.I(passthrough, self.data.df['prev_asia_low'].values, name="prev_asia_low")
        self.is_swing_high = self.I(passthrough, self.data.df['is_swing_high'].values, name="is_swing_high")
        self.is_swing_low = self.I(passthrough, self.data.df['is_swing_low'].values, name="is_swing_low")
        self.hour = self.I(passthrough, self.data.df['hour'].values, name="hour")

        self.reset_state()

    def next(self):
        current_bar = len(self.data) - 1
        current_price = self.data.Close[-1]

        # Time-based exit: Close all positions before the end of the day (e.g., 23:00 UTC)
        if self.position and self.hour[-1] == 23:
            self.position.close()
            self.reset_state()
            return

        if self.position:
            return

        if self.state == State.WAITING_FOR_SWIPE:
            if self.is_london[-1]:
                if self.data.High[-1] > self.prev_asia_high[-1] and self.is_swing_high[-1]:
                    self.state = State.M_PEAK_ONE
                    self.pattern_start_bar = current_bar
                    self.m_peak_one_price = self.data.High[-1]
                elif self.data.Low[-1] < self.prev_asia_low[-1] and self.is_swing_low[-1]:
                    self.state = State.W_TROUGH_ONE
                    self.pattern_start_bar = current_bar
                    self.w_trough_one_price = self.data.Low[-1]

        elif self.state == State.M_PEAK_ONE:
            if self.is_swing_low[-1]:
                self.state = State.M_CENTER_TROUGH
            elif current_bar - self.pattern_start_bar > self.m_pattern_max_bars:
                self.reset_state()

        elif self.state == State.M_CENTER_TROUGH:
            if self.is_swing_high[-1]:
                self.state = State.CHECK_CONSOLIDATION_M
                self.pattern_hod = max(self.m_peak_one_price, self.data.High[-1])
                self.consolidation_start_bar = current_bar
            elif current_bar - self.pattern_start_bar > self.m_pattern_max_bars:
                self.reset_state()

        elif self.state == State.CHECK_CONSOLIDATION_M:
            if current_bar - self.consolidation_start_bar >= self.consolidation_bars:
                consolidation_window = self.data.df.iloc[self.consolidation_start_bar:current_bar+1]
                price_range = consolidation_window['High'].max() - consolidation_window['Low'].min()
                if (price_range / self.pattern_hod) < self.consolidation_range_pct:
                    sl = self.pattern_hod * (1 + self.sl_buffer_pct)
                    # Ensure entry price is below stop-loss for a short
                    if current_price < sl:
                        risk = sl - current_price
                        tp = current_price - (risk * self.risk_reward_ratio)
                        if tp > 0:
                            self.sell(sl=sl, tp=tp)
                self.reset_state()

        elif self.state == State.W_TROUGH_ONE:
            if self.is_swing_high[-1]:
                self.state = State.W_CENTER_PEAK
            elif current_bar - self.pattern_start_bar > self.w_pattern_max_bars:
                self.reset_state()

        elif self.state == State.W_CENTER_PEAK:
            if self.is_swing_low[-1]:
                self.state = State.CHECK_CONSOLIDATION_W
                self.pattern_lod = min(self.w_trough_one_price, self.data.Low[-1])
                self.consolidation_start_bar = current_bar
            elif current_bar - self.pattern_start_bar > self.w_pattern_max_bars:
                self.reset_state()

        elif self.state == State.CHECK_CONSOLIDATION_W:
            if current_bar - self.consolidation_start_bar >= self.consolidation_bars:
                consolidation_window = self.data.df.iloc[self.consolidation_start_bar:current_bar+1]
                price_range = consolidation_window['High'].max() - consolidation_window['Low'].min()
                if (price_range / self.pattern_lod) < self.consolidation_range_pct:
                    sl = self.pattern_lod * (1 - self.sl_buffer_pct)
                    # Ensure entry price is above stop-loss for a long
                    if current_price > sl:
                        risk = current_price - sl
                        tp = current_price + (risk * self.risk_reward_ratio)
                        if tp > 0:
                            self.buy(sl=sl, tp=tp)
                self.reset_state()

    def reset_state(self):
        self.state = State.WAITING_FOR_SWIPE
        self.pattern_start_bar = None
        self.m_peak_one_price = None
        self.m_center_trough_price = None
        self.w_trough_one_price = None
        self.w_center_peak_price = None

if __name__ == '__main__':
    import os
    import json

    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.strip().title() for c in data.columns]
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Preprocess the data
        data = preprocess_data(data)

    else:
        print(f"Data file not found at {data_path}. Generating synthetic data.")
        from backtesting.test import GOOG
        data = GOOG.iloc[-2000:].copy()
        # Also preprocess synthetic data to ensure columns exist
        data = preprocess_data(data)


    bt = Backtest(data, MarketMakerCycleReversalStrategy, cash=100_000, commission=.002)

    stats = bt.run()

    print(stats)

    # Save results
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame, Strategy)):
                continue
            elif pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, np.integer):
                sanitized[key] = int(value)
            elif isinstance(value, np.floating):
                sanitized[key] = float(value)
            elif isinstance(value, pd.Timestamp):
                sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                 sanitized[key] = str(value)
            else:
                sanitized[key] = value
        return sanitized

    clean_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(clean_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    try:
        bt.plot(filename='results/market_maker_cycle_reversal.html')
    except Exception as e:
        print(f"Could not generate plot: {e}")
