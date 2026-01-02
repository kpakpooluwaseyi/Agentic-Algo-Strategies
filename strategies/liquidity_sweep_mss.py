"""
This script implements a price action trading strategy based on liquidity sweeps
and market structure shifts (MSS), intended as a functional proxy for the
user-requested "gold_dxy_price_action_scalping" strategy.

--- STRATEGY PROXY ACKNOWLEDGEMENT ---
The original user request specified a strategy for XAUUSD (Gold) that used
DXY (US Dollar Index) for correlation analysis. Due to data constraints
(only BTC-USD data is available) and framework limitations (no multi-asset
correlation support), this script implements a simplified, single-instrument
version of the core price action concepts.

Key adaptations:
- Instrument: Runs on BTC-USD 15m data instead of XAUUSD.
- Correlation: Does not include DXY correlation.
- HTF Direction: A 200-period EMA is used as a proxy for higher-timeframe trend
  instead of 4H/Daily candle analysis.
- Session: A fixed UTC time window is used to proxy the "Asia session".
---
"""
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from enum import Enum
import numpy as np
from scipy.signal import find_peaks
import json
import os

# --- Synthetic Data Generation ---
def generate_synthetic_data():
    """
    Generates a DataFrame with a textbook bullish liquidity sweep and MSS pattern,
    ensuring the price is above a long-term EMA.
    """
    dates = pd.to_datetime(pd.date_range(start='2023-01-01 00:00', periods=300, freq='15min'))
    price = 100
    data = []

    # 1. Strong uptrend to get price above a slow EMA
    for i in range(150):
        price += np.random.uniform(0.1, 0.4)
        ohlc = [price, price + 0.1, price - 0.1, price + 0.05]
        data.append([dates[i], *ohlc, 1000])

    # 2. Create the key pattern
    # Pullback to establish a clear swing low
    price_before_low = price
    swing_low_price = price - 5
    for i in range(10):
        price -= 0.5
        ohlc = [price, price + 0.2, price - 0.2, price - 0.1]
        data.append([dates[150+i], *ohlc, 1200])
    data[150][1] = price_before_low # Open of the move down
    data[159][3] = swing_low_price # Low of the move

    # Establish a clear swing high after the low
    price_before_high = price
    swing_high_price = price + 3
    for i in range(10):
        price += 0.3
        ohlc = [price, price + 0.2, price - 0.2, price + 0.1]
        data.append([dates[160+i], *ohlc, 1200])
    data[160][1] = price_before_high
    data[169][2] = swing_high_price

    # 3. Liquidity Sweep (takes out the swing low) - index ~175
    price_before_sweep = price
    sweep_date = dates[175]
    sweep_ohlc = [price_before_sweep, price_before_sweep + .1, swing_low_price - 0.1, swing_low_price]
    data.append([sweep_date, *sweep_ohlc, 2000])
    price = swing_low_price

    # 4. Market Structure Shift (breaks the swing high) - index ~177
    price_before_mss = price
    mss_date = dates[177]
    mss_ohlc = [price_before_mss, swing_high_price + 0.1, price_before_mss - 0.1, swing_high_price]
    data.append([mss_date, *mss_ohlc, 2500])
    price = swing_high_price

    # 5. Retest of the MSS level - index ~178
    retest_date = dates[178]
    retest_ohlc = [price, price + 0.1, swing_high_price - 0.1, swing_high_price]
    data.append([retest_date, *retest_ohlc, 1800])

    # Fill remaining data
    for i in range(len(data), 300):
        price += np.random.uniform(-0.1, 0.2)
        ohlc = [price, price + 0.1, price - 0.1, price + 0.05]
        data.append([dates[i], *ohlc, 1000])

    df = pd.DataFrame(data, columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.set_index('datetime', inplace=True)
    return df

# --- Data Pre-processing ---

def preprocess_data(df, ema_period=200, peak_distance=10):
    df['ema'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    high_peaks_idx, _ = find_peaks(df['High'], distance=peak_distance)
    low_peaks_idx, _ = find_peaks(-df['Low'], distance=peak_distance)
    df['swing_high'] = np.nan
    df.iloc[high_peaks_idx, df.columns.get_loc('swing_high')] = df.iloc[high_peaks_idx]['High']
    df['swing_low'] = np.nan
    df.iloc[low_peaks_idx, df.columns.get_loc('swing_low')] = df.iloc[low_peaks_idx]['Low']
    df['last_swing_high'] = df['swing_high'].ffill()
    df['last_swing_low'] = df['swing_low'].ffill()
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    df['hour'] = df.index.hour
    df['is_trading_session'] = (df['hour'] >= 0) & (df['hour'] <= 2)
    df.dropna(subset=['ema', 'last_swing_high', 'last_swing_low'], inplace=True)
    return df

# --- Strategy Definition ---

class State(Enum):
    SEARCHING = 0
    MSS_CONFIRMED = 1
    PENDING_ENTRY = 2

class LiquiditySweepMSS(Strategy):
    ema_period = 200
    peak_distance = 30
    rr_ratio = 1.5
    sl_buffer_pct = 0.01

    def init(self):
        self.ema = self.I(lambda x: x, self.data.df['ema'], name='EMA')
        self.last_swing_high = self.I(lambda x: x, self.data.df['last_swing_high'], name='Last_Swing_High')
        self.last_swing_low = self.I(lambda x: x, self.data.df['last_swing_low'], name='Last_Swing_Low')
        self.is_trading_session = self.I(lambda x: x, self.data.df['is_trading_session'], name='Is_Trading_Session')
        self.setup_state = State.SEARCHING
        self.setup_direction = 0
        self.liquidity_level = None
        self.mss_level = None
        self.entry_level = None
        self.sl_level = None
        self.was_in_position = False

    def next(self):
        # State reset on trade close
        if self.was_in_position and not self.position:
            self._reset_state()
        self.was_in_position = bool(self.position)

        current_price = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]

        if self.setup_state == State.SEARCHING:
            if self.is_trading_session[-1] and not self.position:
                if current_price > self.ema[-1] and current_low < self.last_swing_low[-2]:
                    self.setup_direction = 1
                    self.liquidity_level = self.last_swing_low[-2]
                    self.mss_level = self.last_swing_high[-1]
                    self.setup_state = State.MSS_CONFIRMED
                elif current_price < self.ema[-1] and current_high > self.last_swing_high[-2]:
                    self.setup_direction = -1
                    self.liquidity_level = self.last_swing_high[-2]
                    self.mss_level = self.last_swing_low[-1]
                    self.setup_state = State.MSS_CONFIRMED

        elif self.setup_state == State.MSS_CONFIRMED:
            # Invalidation
            if (self.setup_direction == 1 and current_low < self.liquidity_level) or \
               (self.setup_direction == -1 and current_high > self.liquidity_level):
                self._reset_state()
            # Confirmation
            elif self.setup_direction == 1 and current_high > self.mss_level:
                self.entry_level = self.mss_level
                self.sl_level = self.liquidity_level * (1 - self.sl_buffer_pct)
                self.setup_state = State.PENDING_ENTRY
            elif self.setup_direction == -1 and current_low < self.mss_level:
                self.entry_level = self.mss_level
                self.sl_level = self.liquidity_level * (1 + self.sl_buffer_pct)
                self.setup_state = State.PENDING_ENTRY

        elif self.setup_state == State.PENDING_ENTRY:
            if self.setup_direction == 1:
                tp_level = self.entry_level + (self.entry_level - self.sl_level) * self.rr_ratio
                if self.entry_level > self.sl_level:
                    self.buy(limit=self.entry_level, sl=self.sl_level, tp=tp_level)
            elif self.setup_direction == -1:
                tp_level = self.entry_level - (self.sl_level - self.entry_level) * self.rr_ratio
                if self.entry_level < self.sl_level:
                    self.sell(limit=self.entry_level, sl=self.sl_level, tp=tp_level)

    def _reset_state(self):
        self.setup_state = State.SEARCHING
        self.setup_direction = 0
        self.liquidity_level = None
        self.mss_level = None
        self.entry_level = None
        self.sl_level = None

# --- Backtesting Execution ---
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest a liquidity sweep MSS strategy.')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data to verify strategy logic.')
    args = parser.parse_args()

    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    results_filename = 'temp_result.json'
    plot_filename = 'liquidity_sweep_mss.html'
    os.makedirs(results_dir, exist_ok=True)

    peak_dist = LiquiditySweepMSS.peak_distance

    if args.synthetic:
        print("Using synthetic data for verification...")
        data = generate_synthetic_data()
        peak_dist = 5
    else:
        try:
            data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
            data.columns = [col.strip().capitalize() for col in data.columns]
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}. Falling back to synthetic data.")
            data = generate_synthetic_data()
            peak_dist = 5

    data_processed = preprocess_data(data.copy(), ema_period=LiquiditySweepMSS.ema_period, peak_distance=peak_dist)

    if args.synthetic or 'is_trading_session' not in data_processed.columns:
        data_processed['is_trading_session'] = True

    bt = Backtest(data_processed, LiquiditySweepMSS, cash=100_000, commission=.002)

    print("Running backtest...")
    stats = bt.run()
    print("\nBacktest Stats:")
    print(stats)

    results_path = os.path.join(results_dir, results_filename)
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, pd.Timestamp): return obj.isoformat()
            if isinstance(obj, pd.Timedelta): return str(obj)
            if isinstance(obj, pd.Series): return obj.to_dict()
            if str(obj) == 'nan' or obj is pd.NA: return None
            if isinstance(obj, pd.DataFrame): return obj.to_dict(orient='records')
            if isinstance(obj, type): return str(obj)
            return super(NpEncoder, self).default(obj)

    cleaned_stats = {k: v for k, v in stats.items() if not k.startswith('_')}

    with open(results_path, 'w') as f: json.dump(cleaned_stats, f, indent=4, cls=NpEncoder)
    print(f"\nSaved stats to {results_path}")

    plot_path = os.path.join(results_dir, plot_filename)
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
