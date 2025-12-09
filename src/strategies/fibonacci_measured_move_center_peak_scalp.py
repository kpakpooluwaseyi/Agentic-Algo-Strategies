from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os

# +-----------------------------------------------------------------------------+
# | C A N D L E S T I C K   P A T T E R N   H E L P E R S                       |
# +-----------------------------------------------------------------------------+

def is_bearish_engulfing(df, i):
    """Detects a bearish engulfing pattern at index i."""
    if i < 1:
        return False
    current = df.iloc[i]
    previous = df.iloc[i - 1]

    if not (isinstance(current, pd.Series) and isinstance(previous, pd.Series)):
         return False

    if 'Close' not in current or 'Open' not in current or 'Close' not in previous or 'Open' not in previous:
        return False

    # Previous candle must be bullish
    if previous['Close'] <= previous['Open']:
        return False

    # Current candle must be bearish
    if current['Close'] >= current['Open']:
        return False

    # Current candle must engulf the previous one
    if current['Open'] > previous['Close'] and current['Close'] < previous['Open']:
        return True

    return False

def is_shooting_star(df, i):
    """Detects a shooting star pattern at index i."""
    if i < 1:
        return False

    candle = df.iloc[i]
    if not isinstance(candle, pd.Series):
        return False

    if 'Open' not in candle or 'Close' not in candle or 'High' not in candle or 'Low' not in candle:
        return False

    body_size = abs(candle['Open'] - candle['Close'])
    upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
    lower_wick = min(candle['Open'], candle['Close']) - candle['Low']

    if body_size > 0 and upper_wick > 2 * body_size and lower_wick < body_size / 2:
        return True

    return False

# +-----------------------------------------------------------------------------+
# | S T R A T E G Y   C L A S S                                                 |
# +-----------------------------------------------------------------------------+

class FibonacciMeasuredMoveCenterPeakScalpStrategy(Strategy):
    min_rr = 3.0
    sl_padding_pct = 0.01
    aoi_tolerance_pct = 0.02

    def init(self):
        self.setup_high = self.data.df['setup_high']
        self.setup_low = self.data.df['setup_low']
        self.aoi_50 = self.data.df['aoi_50']

    def next(self):
        if self.position:
            return

        if np.isnan(self.setup_high.iloc[-1]):
            return

        aoi_upper_bound = self.aoi_50.iloc[-1] * (1 + self.aoi_tolerance_pct)
        aoi_lower_bound = self.aoi_50.iloc[-1] * (1 - self.aoi_tolerance_pct)

        price = self.data.Close[-1]

        if not (aoi_lower_bound <= price <= aoi_upper_bound):
            return

        current_index = len(self.data.df) - 1
        is_reversal = (is_bearish_engulfing(self.data.df, current_index) or
                       is_shooting_star(self.data.df, current_index))

        if not is_reversal:
            return

        center_peak_high = self.data.High[-1]
        sl = center_peak_high * (1 + self.sl_padding_pct)
        initial_drop_low = self.setup_low.iloc[-1]
        measured_move_range = center_peak_high - initial_drop_low
        tp = center_peak_high - (measured_move_range * 0.5)

        risk = abs(sl - price)
        reward = abs(price - tp)

        if risk <= 0:
            return

        rr = reward / risk

        if rr >= self.min_rr:
            self.sell(sl=sl, tp=tp)

# +-----------------------------------------------------------------------------+
# | D A T A   P R E P A R A T I O N   &   B A C K T E S T I N G                 |
# +-----------------------------------------------------------------------------+

def generate_synthetic_data():
    """
    Generates a clean, unmistakable synthetic 1M OHLC dataset for the strategy.
    This pattern is designed to be perfectly picked up by the preprocessing logic.
    """
    periods = 1000
    dates = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.arange(periods), 'm')
    df = pd.DataFrame(index=dates)

    # 1. Initial stable price
    price = 110
    df['Open'] = price
    df['High'] = price
    df['Low'] = price
    df['Close'] = price

    # 2. Create the significant 15M+ drop
    drop_high = 110
    drop_low = 90
    drop_start_idx, drop_end_idx = 100, 130
    drop_prices = np.linspace(drop_high, drop_low, drop_end_idx - drop_start_idx)
    df.iloc[drop_start_idx:drop_end_idx, df.columns.get_loc('Open')] = drop_prices + 0.1
    df.iloc[drop_start_idx:drop_end_idx, df.columns.get_loc('High')] = drop_prices + 0.2
    df.iloc[drop_start_idx:drop_end_idx, df.columns.get_loc('Low')] = drop_prices - 0.2
    df.iloc[drop_start_idx:drop_end_idx, df.columns.get_loc('Close')] = drop_prices

    # 3. Consolidate at the low
    df.iloc[drop_end_idx:300] = drop_low

    # 4. Clean retracement to the 50% AOI
    aoi_50 = drop_high - (drop_high - drop_low) * 0.5  # Exactly 100
    retrace_start_idx, retrace_end_idx = 300, 350
    retrace_prices = np.linspace(drop_low, aoi_50, retrace_end_idx - retrace_start_idx)
    df.iloc[retrace_start_idx:retrace_end_idx, df.columns.get_loc('Open')] = retrace_prices - 0.1
    df.iloc[retrace_start_idx:retrace_end_idx, df.columns.get_loc('High')] = retrace_prices + 0.1
    df.iloc[retrace_start_idx:retrace_end_idx, df.columns.get_loc('Low')] = retrace_prices - 0.2
    df.iloc[retrace_start_idx:retrace_end_idx, df.columns.get_loc('Close')] = retrace_prices

    # 5. Textbook Bearish Engulfing candle at the AOI
    idx = retrace_end_idx
    # Previous bullish candle
    df.iloc[idx-1, df.columns.get_loc('Open')] = 99.8
    df.iloc[idx-1, df.columns.get_loc('Close')] = 100.0
    df.iloc[idx-1, df.columns.get_loc('High')] = 100.05
    df.iloc[idx-1, df.columns.get_loc('Low')] = 99.75
    # The bearish engulfing candle
    df.iloc[idx, df.columns.get_loc('Open')] = 100.1
    df.iloc[idx, df.columns.get_loc('Close')] = 99.7
    df.iloc[idx, df.columns.get_loc('High')] = 100.15
    df.iloc[idx, df.columns.get_loc('Low')] = 99.65

    # 6. Drop towards the take-profit target
    tp_target = 100.15 - ((100.15 - drop_low) * 0.5)
    drop2_start_idx, drop2_end_idx = idx + 1, idx + 51
    drop2_prices = np.linspace(99.7, tp_target, drop2_end_idx - drop2_start_idx)
    df.iloc[drop2_start_idx:drop2_end_idx, df.columns.get_loc('Close')] = drop2_prices
    df.iloc[drop2_start_idx:drop2_end_idx, df.columns.get_loc('Open')] = drop2_prices + 0.1

    # Fill NaNs from partial assignments
    df.ffill(inplace=True)

    return df

def preprocess_data(df_1m, lookback_period=5, drop_pct_threshold=0.08):
    """Identifies 15M setups and merges them into the 1M data using a vectorized approach."""
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()

    df_15m['rolling_high'] = df_15m['High'].rolling(window=lookback_period).max()
    df_15m['rolling_low'] = df_15m['Low'].rolling(window=lookback_period).min()
    df_15m['pct_drop'] = (df_15m['rolling_high'] - df_15m['Low']) / df_15m['rolling_high']

    is_setup = df_15m['pct_drop'] > drop_pct_threshold

    df_15m['setup_high'] = np.nan
    df_15m['setup_low'] = np.nan
    df_15m.loc[is_setup, 'setup_high'] = df_15m.loc[is_setup, 'rolling_high']
    df_15m.loc[is_setup, 'setup_low'] = df_15m.loc[is_setup, 'rolling_low']

    df_15m[['setup_high', 'setup_low']] = df_15m[['setup_high', 'setup_low']].ffill()
    df_15m['aoi_50'] = df_15m['setup_high'] - (df_15m['setup_high'] - df_15m['setup_low']) * 0.5

    df_merged = pd.merge_asof(
        df_1m, df_15m[['setup_high', 'setup_low', 'aoi_50']],
        left_index=True, right_index=True, direction='backward'
    )
    return df_merged

if __name__ == '__main__':
    data_1m = generate_synthetic_data()
    data_processed = preprocess_data(data_1m)

    bt = Backtest(data_processed, FibonacciMeasuredMoveCenterPeakScalpStrategy, cash=10000, commission=.002)

    stats = bt.optimize(
        min_rr=np.arange(2.0, 5.5, 0.5).tolist(),
        sl_padding_pct=np.arange(0.005, 0.03, 0.005).tolist(),
        aoi_tolerance_pct=np.arange(0.01, 0.05, 0.01).tolist(),
        maximize='Sharpe Ratio'
    )

    print("Best Run Stats:")
    print(stats)

    os.makedirs('results', exist_ok=True)
    if stats['# Trades'] > 0:
        stats_dict = {
            'strategy_name': 'fibonacci_measured_move_center_peak_scalp',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }
    else:
        stats_dict = {
            'strategy_name': 'fibonacci_measured_move_center_peak_scalp',
            'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0,
            'win_rate': 0.0, 'total_trades': 0
        }
    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print("\nResults saved to results/temp_result.json")

    bt.plot(filename="results/fibonacci_strategy_plot.html", open_browser=False)
    print("Plot saved to results/fibonacci_strategy_plot.html")
