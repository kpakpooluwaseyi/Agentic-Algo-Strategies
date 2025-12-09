from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks

class FibonacciCenterPeakScalpStrategy(Strategy):
    ema_period = 50
    rr_ratio = 5
    confirmation_window = 3

    def init(self):
        # Indicators for 1M data
        self.ema = self.I(lambda x: pd.Series(x).ewm(span=self.ema_period, adjust=False).mean(), self.data.Close)

        # Pre-processed 15M data
        self.peak_price_15m = self.I(lambda: self.data.df['peak_price_15m'], name='peak_price_15m')
        self.trough_price_15m = self.I(lambda: self.data.df['trough_price_15m'], name='trough_price_15m')
        self.swing_high_t_15m = self.I(lambda: self.data.df['swing_high_t_15m'].astype('datetime64[ns]').astype(np.int64), name='swing_high_t_15m')
        self.swing_low_t_15m = self.I(lambda: self.data.df['swing_low_t_15m'].astype('datetime64[ns]').astype(np.int64), name='swing_low_t_15m')

        # Consolidated state machine variables
        self.last_swing_t = {'long': None, 'short': None}
        self.level_start = {'long': None, 'short': None}
        self.level_end = {'long': None, 'short': None}
        self.center_extreme = {'long': None, 'short': None}
        self.setup_valid = {'long': False, 'short': False}
        self.looking_for_trade = {'long': False, 'short': False}

    def _check_entry_conditions(self, trade_type):
        current_price = self.data.Close[-1]
        is_long = trade_type == 'long'

        # Determine current swing times
        current_swing_start_t = self.swing_low_t_15m[-1] if is_long else self.swing_high_t_15m[-1]
        current_swing_end_t = self.swing_high_t_15m[-1] if is_long else self.swing_low_t_15m[-1]

        # 1. Detect a new swing to start a potential setup
        if current_swing_start_t and current_swing_start_t != self.last_swing_t[trade_type]:
            self.level_start[trade_type] = self.trough_price_15m[-1] if is_long else self.peak_price_15m[-1]
            self.last_swing_t[trade_type] = current_swing_start_t
            self.setup_valid[trade_type] = False
            self.looking_for_trade[trade_type] = False

        # 2. Detect a subsequent opposite swing to confirm the pattern
        if self.level_start[trade_type] and not self.setup_valid[trade_type] and current_swing_end_t and current_swing_end_t != self.last_swing_t['short' if is_long else 'long']:
            self.level_end[trade_type] = self.peak_price_15m[-1] if is_long else self.trough_price_15m[-1]
            self.last_swing_t['short' if is_long else 'long'] = current_swing_end_t
            if (is_long and self.level_end[trade_type] > self.level_start[trade_type]) or \
               (not is_long and self.level_end[trade_type] < self.level_start[trade_type]):
                self.setup_valid[trade_type] = True

        # 3. If setup is valid, wait for price to enter AOI or invalidate
        if self.setup_valid[trade_type]:
            invalidation_price = self.level_start[trade_type]
            if (is_long and current_price < invalidation_price) or (not is_long and current_price > invalidation_price):
                self.setup_valid[trade_type] = False
                self.looking_for_trade[trade_type] = False
                self.level_start[trade_type] = None
            else:
                fib_1_50 = self.level_start[trade_type] + (self.level_end[trade_type] - self.level_start[trade_type]) * 0.5
                if not self.looking_for_trade[trade_type] and \
                   ((is_long and current_price < fib_1_50) or (not is_long and current_price > fib_1_50)):
                    self.looking_for_trade[trade_type] = True
                    self.center_extreme[trade_type] = self.data.Low[-1] if is_long else self.data.High[-1]

        # 4. If in AOI, look for confirmation candle and enter trade
        if self.looking_for_trade[trade_type] and not self.position:
            self.center_extreme[trade_type] = min(self.center_extreme[trade_type], self.data.Low[-1]) if is_long else max(self.center_extreme[trade_type], self.data.High[-1])

            # Confluence Check with EMA
            ema_confluence = (is_long and current_price > self.ema[-1]) or (not is_long and current_price < self.ema[-1])

            # Confirmation Check within a window
            confirmation_found = False
            for i in range(self.confirmation_window):
                if (is_long and self.data.Close[-1-i] > self.data.Open[-1-i]) or \
                   (not is_long and self.data.Close[-1-i] < self.data.Open[-1-i]):
                    confirmation_found = True
                    break

            if ema_confluence and confirmation_found:
                sl = self.center_extreme[trade_type] * (0.999 if is_long else 1.001)
                tp = self.center_extreme[trade_type] + (self.level_end[trade_type] - self.center_extreme[trade_type]) * 0.5

                rr = (abs(tp - current_price) / abs(sl - current_price)) if abs(sl-current_price) > 0 else 0
                if rr >= self.rr_ratio:
                    if is_long: self.buy(sl=sl, tp=tp)
                    else: self.sell(sl=sl, tp=tp)
                    self.setup_valid[trade_type] = False
                    self.looking_for_trade[trade_type] = False
                    self.level_start[trade_type] = None

    def next(self):
        # Check entry conditions for both trade types
        self._check_entry_conditions('long')
        self._check_entry_conditions('short')

if __name__ == '__main__':
    def generate_synthetic_data(periods=2000, m_pattern=True):
        """Generates synthetic price data with a clear M or W pattern."""
        np.random.seed(42)
        time_index = pd.date_range(start='2023-01-01', periods=periods, freq='min')
        price = 100
        data = []

        # Base trend
        base = price + np.cumsum(np.random.randn(periods) * 0.05)

        if m_pattern:
            # M-Pattern (for short setup)
            # Level Drop
            base[100:200] -= np.linspace(0, 10, 100)
            # Retracement (Center Peak)
            base[200:300] += np.linspace(0, 6, 100) # Retraces past 50%
            # Second Drop
            base[300:400] -= np.linspace(0, 12, 100)
        else:
            # W-Pattern (for long setup)
            # Level Rise
            base[100:200] += np.linspace(0, 10, 100)
            # Retracement (Center Trough)
            base[200:300] -= np.linspace(0, 6, 100) # Retraces past 50%
            # Second Rise
            base[300:400] += np.linspace(0, 12, 100)

        df = pd.DataFrame({'Close': base}, index=time_index)
        df['Open'] = df['Close'].shift(1).fillna(method='bfill')
        df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 0.2, size=len(df))
        df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 0.2, size=len(df))
        return df

    # Generate synthetic data for M-pattern (short trade)
    # Switch to `m_pattern=False` to test the long logic
    data = generate_synthetic_data(m_pattern=False)

    def preprocess_data(df_1m):
        """
        Resamples 1M data to 15M, identifies swing points, and merges the context back.
        """
        df_15m = df_1m['Close'].resample('15min').ohlc().dropna()

        # Find peaks (swing highs) and troughs (swing lows) on the 15M timeframe
        peak_indices, _ = find_peaks(df_15m['high'], prominence=1, distance=3)
        trough_indices, _ = find_peaks(-df_15m['low'], prominence=1, distance=3)

        df_15m['peak_price'] = np.nan
        df_15m['trough_price'] = np.nan
        df_15m.iloc[peak_indices, df_15m.columns.get_loc('peak_price')] = df_15m['high'].iloc[peak_indices]
        df_15m.iloc[trough_indices, df_15m.columns.get_loc('trough_price')] = df_15m['low'].iloc[trough_indices]

        df_15m['swing_high_t'] = np.nan
        df_15m['swing_low_t'] = np.nan
        df_15m.iloc[peak_indices, df_15m.columns.get_loc('swing_high_t')] = df_15m.index[peak_indices]
        df_15m.iloc[trough_indices, df_15m.columns.get_loc('swing_low_t')] = df_15m.index[trough_indices]

        # Forward fill to carry the last swing point forward
        df_15m.ffill(inplace=True)

        # Merge the 15M context into the 1M data
        df_merged = pd.merge_asof(df_1m, df_15m[['peak_price', 'trough_price', 'swing_high_t', 'swing_low_t']],
                                  left_index=True, right_index=True, direction='backward')
        df_merged.rename(columns={'peak_price': 'peak_price_15m', 'trough_price': 'trough_price_15m',
                                  'swing_high_t': 'swing_high_t_15m', 'swing_low_t': 'swing_low_t_15m'}, inplace=True)
        return df_merged

    data = preprocess_data(data)

    import os
    os.makedirs('results', exist_ok=True)

    # Save preprocessed data for debugging
    data.to_csv('results/preprocessed_data.csv')

    bt = Backtest(data, FibonacciCenterPeakScalpStrategy, cash=100_000, commission=.002)

    stats = bt.optimize(
        ema_period=range(20, 101, 20),
        rr_ratio=range(3, 8),
        confirmation_window=range(2, 6),
        maximize='Sharpe Ratio',
    )

    print(stats)

    def _sanitize_stats(stats_series):
        sanitized = {
            'strategy_name': 'fibonacci_center_peak_scalp',
            'return': float(stats_series.get('Return [%]', 0.0)),
            'sharpe': float(stats_series.get('Sharpe Ratio', 0.0)),
            'max_drawdown': float(stats_series.get('Max. Drawdown [%]', 0.0)),
            'win_rate': float(stats_series.get('Win Rate [%]', 0.0)),
            'total_trades': int(stats_series.get('# Trades', 0))
        }
        if np.isnan(sanitized['sharpe']):
            sanitized['sharpe'] = 0.0
        if np.isnan(sanitized['win_rate']):
            sanitized['win_rate'] = 0.0
        return sanitized

    with open('results/temp_result.json', 'w') as f:
        json.dump(_sanitize_stats(stats), f, indent=2)

    try:
        bt.plot(filename="results/fibonacci_center_peak_scalp.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
