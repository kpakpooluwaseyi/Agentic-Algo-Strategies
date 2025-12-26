
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

# ============== UTILITY AND PRE-PROCESSING FUNCTIONS ==============

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds session information, historical data points, and swing points to the DataFrame.
    """
    # Ensure index is a DatetimeIndex
    df.index = pd.to_datetime(df.index)

    # Define session times in UTC
    asia_start_hour = 0
    asia_end_hour = 8
    london_start_hour = 8
    london_end_hour = 16

    df['hour'] = df.index.hour
    df['day'] = df.index.date

    # Identify sessions
    df['is_asia'] = (df['hour'] >= asia_start_hour) & (df['hour'] < asia_end_hour)
    df['is_london'] = (df['hour'] >= london_start_hour) & (df['hour'] < london_end_hour)

    # --- Calculate Asia Session Levels ---
    asia_high_map = df[df['is_asia']].groupby('day')['High'].max()
    asia_low_map = df[df['is_asia']].groupby('day')['Low'].min()

    df['asia_high'] = df['day'].map(asia_high_map)
    df['asia_low'] = df['day'].map(asia_low_map)

    # Forward fill the session data, back-filling first to handle missing initial values
    df['asia_high'] = df['asia_high'].bfill().ffill()
    df['asia_low'] = df['asia_low'].bfill().ffill()

    df['asia_range'] = df['asia_high'] - df['asia_low']
    df['asia_range_pct'] = (df['asia_range'] / df['asia_low']) * 100

    # --- Find Swing Points ---
    # Further relaxed peak finding for higher sensitivity
    high_peaks, _ = find_peaks(df['High'], distance=3, prominence=df['High'].std() * 0.1)
    low_peaks, _ = find_peaks(-df['Low'], distance=3, prominence=df['Low'].std() * 0.1)

    df['swing_high'] = False
    df['swing_low'] = False
    df.iloc[high_peaks, df.columns.get_loc('swing_high')] = True
    df.iloc[low_peaks, df.columns.get_loc('swing_low')] = True

    # --- Add Confluence Indicators ---
    df.ta.ema(length=5, append=True)
    df.ta.ema(length=13, append=True)

    # Drop rows where essential session data is missing
    df.dropna(subset=['asia_high', 'asia_low', 'EMA_5', 'EMA_13'], inplace=True)
    df.drop(columns=['hour', 'day'], inplace=True, errors='ignore')
    return df

# ============== STRATEGY DEFINITION ==============

class SecondLegMOrWReversalStrategy(Strategy):
    """
    Trades M- and W-pattern reversals that form after a liquidity grab
    above or below the Asian session range.
    """
    # --- Optimization Parameters ---
    # Relaxed for BTC data to allow more frequent setups
    asia_range_max_pct = 5.0
    stop_hunt_pct = 0.01      # Minimum % break of Asia range to qualify as a stop hunt
    sl_buffer_pct = 0.1       # % buffer for stop loss placement
    risk_reward_ratio = 2.0   # Desired risk:reward ratio
    max_bars_between_legs = 6 # 90 minutes (6 bars * 15 min)
    time_exit_bars = 8        # 2 hours (8 bars * 15 min)

    def init(self):
        # Pre-calculate access to columns for performance
        self.is_london = self.data.is_london.astype(bool)
        self.asia_high = self.data.asia_high
        self.asia_low = self.data.asia_low
        self.asia_range_pct = self.data.asia_range_pct
        self.swing_high = self.data.swing_high.astype(bool)
        self.swing_low = self.data.swing_low.astype(bool)
        self.ema_5 = self.data.EMA_5
        self.ema_13 = self.data.EMA_13

        # State machine variables
        self.m_peak_1_price = None
        self.m_peak_1_idx = None
        self.w_trough_1_price = None
        self.w_trough_1_idx = None
        self.stop_hunt_active_high = False
        self.stop_hunt_active_low = False

    def next(self):
        # --- Trade Management ---
        if self.position:
            # Time-based exit
            if self.i - self.trades[0].entry_bar > self.time_exit_bars:
                self.position.close()

        # --- Reset Logic ---
        # Reset if a new day starts (or London session ends)
        if not self.is_london[-1]:
            self.m_peak_1_price = self.m_peak_1_idx = None
            self.w_trough_1_price = self.w_trough_1_idx = None
            self.stop_hunt_active_high = self.stop_hunt_active_low = False
            return

        # --- Entry Logic ---
        if not self.position:
            is_asia_range_valid = self.asia_range_pct[-1] < self.asia_range_max_pct
            if not is_asia_range_valid:
                return

            # --- M-Pattern (Short) Setup ---
            # 1. Detect Stop Hunt High
            if self.data.High[-1] > self.asia_high[-1] * (1 + self.stop_hunt_pct / 100):
                self.stop_hunt_active_high = True

            # 2. Find First Leg of M
            if self.stop_hunt_active_high and self.swing_high[-1] and self.m_peak_1_price is None:
                self.m_peak_1_price = self.data.High[-1]
                self.m_peak_1_idx = self.i

            # 3. Find Second, Weaker Leg and Enter (or invalidate pattern)
            if self.m_peak_1_price is not None:
                # Invalidation: If price makes a new high, the M-pattern is void. Reset the hunt.
                if self.data.High[-1] > self.m_peak_1_price:
                    self.m_peak_1_price = self.m_peak_1_idx = None
                    self.stop_hunt_active_high = False
                # Confirmation:
                elif self.swing_high[-1] and self.i > self.m_peak_1_idx and self.i - self.m_peak_1_idx <= self.max_bars_between_legs:
                    # Check for bearish EMA crossover
                    ema_crossed_down = self.ema_5[-2] > self.ema_13[-2] and self.ema_5[-1] < self.ema_13[-1]
                    if self.data.High[-1] < self.m_peak_1_price and ema_crossed_down:
                        sl_price = self.m_peak_1_price * (1 + self.sl_buffer_pct / 100)
                        tp_price = self.data.Close[-1] - (sl_price - self.data.Close[-1]) * self.risk_reward_ratio
                        if tp_price < self.data.Close[-1]:
                            self.sell(sl=sl_price, tp=tp_price)
                        self.m_peak_1_price = self.m_peak_1_idx = self.stop_hunt_active_high = False # Reset

            # --- W-Pattern (Long) Setup ---
            # 1. Detect Stop Hunt Low
            if self.data.Low[-1] < self.asia_low[-1] * (1 - self.stop_hunt_pct / 100):
                self.stop_hunt_active_low = True

            # 2. Find First Leg of W
            if self.stop_hunt_active_low and self.swing_low[-1] and self.w_trough_1_price is None:
                self.w_trough_1_price = self.data.Low[-1]
                self.w_trough_1_idx = self.i

            # 3. Find Second, Weaker Leg and Enter (or invalidate pattern)
            if self.w_trough_1_price is not None:
                # Invalidation: If price makes a new low, the W-pattern is void. Reset the hunt.
                if self.data.Low[-1] < self.w_trough_1_price:
                    self.w_trough_1_price = self.w_trough_1_idx = None
                    self.stop_hunt_active_low = False
                # Confirmation:
                elif self.swing_low[-1] and self.i > self.w_trough_1_idx and self.i - self.w_trough_1_idx <= self.max_bars_between_legs:
                    # Check for bullish EMA crossover
                    ema_crossed_up = self.ema_5[-2] < self.ema_13[-2] and self.ema_5[-1] > self.ema_13[-1]
                    if self.data.Low[-1] > self.w_trough_1_price and ema_crossed_up:
                        sl_price = self.w_trough_1_price * (1 - self.sl_buffer_pct / 100)
                        tp_price = self.data.Close[-1] + (self.data.Close[-1] - sl_price) * self.risk_reward_ratio
                        if tp_price > self.data.Close[-1]:
                            self.buy(sl=sl_price, tp=tp_price)
                        self.w_trough_1_price = self.w_trough_1_idx = self.stop_hunt_active_low = False # Reset


# ============== BACKTEST EXECUTION ==============

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        print(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Clean column names: remove whitespace and standardize to title case
        data.columns = [c.strip().title() for c in data.columns]

        print("Preprocessing data...")
        data = preprocess_data(data)

        bt = Backtest(data, SecondLegMOrWReversalStrategy, cash=100_000, commission=.002)

        print("Running backtest...")
        stats = bt.run()
        print(stats)

        # --- Save results to JSON ---
        os.makedirs('results', exist_ok=True)

        # Sanitize stats for JSON by extracting only known, safe metrics
        result_data = {
            'strategy_name': 'second_leg_m_or_w_reversal',
            'Start': stats.get('Start'),
            'End': stats.get('End'),
            'Duration': stats.get('Duration'),
            'Exposure Time [%]': stats.get('Exposure Time [%]'),
            'Equity Final [$]': stats.get('Equity Final [$]'),
            'Equity Peak [$]': stats.get('Equity Peak [$]'),
            'Return [%]': stats.get('Return [%]'),
            'Buy & Hold Return [%]': stats.get('Buy & Hold Return [%]'),
            'Return (Ann.) [%]': stats.get('Return (Ann.) [%]'),
            'Volatility (Ann.) [%]': stats.get('Volatility (Ann.) [%]'),
            'Sharpe Ratio': stats.get('Sharpe Ratio'),
            'Sortino Ratio': stats.get('Sortino Ratio'),
            'Calmar Ratio': stats.get('Calmar Ratio'),
            'Max. Drawdown [%]': stats.get('Max. Drawdown [%]'),
            'Avg. Drawdown [%]': stats.get('Avg. Drawdown [%]'),
            '# Trades': stats.get('# Trades'),
            'Win Rate [%]': stats.get('Win Rate [%]'),
            'Profit Factor': stats.get('Profit Factor'),
            'Expectancy [%]': stats.get('Expectancy [%]'),
        }
        # Convert numpy types to native python types
        for key, value in result_data.items():
            if isinstance(value, (np.int64, np.int32)):
                result_data[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                result_data[key] = float(value) if not np.isnan(value) else None
            elif isinstance(value, pd.Timestamp):
                result_data[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                result_data[key] = str(value)

        with open('results/temp_result.json', 'w') as f:
            json.dump(result_data, f, indent=4)

        print("\nResults saved to results/temp_result.json")

        # --- Generate plot ---
        plot_filename = 'results/_second_leg_m_or_w_reversal_.html'
        print(f"Generating plot... {plot_filename}")
        try:
            bt.plot(filename=plot_filename, open_browser=False)
            print("Plot saved successfully.")
        except Exception as e:
            print(f"Could not generate plot: {e}")
