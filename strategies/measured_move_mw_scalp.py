from enum import Enum
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from backtesting import Backtest, Strategy

# --- Configuration ---
INITIAL_CASH = 100_000
COMMISSION_PCT = 0.002
RISK_PER_TRADE_PCT = 0.01  # 1% of equity

# --- State Management ---
class TradeState(Enum):
    SEARCHING = 1
    IN_AOI = 2

# --- Data Generation ---
def generate_m_pattern_data(n_points=3000):
    """Generates synthetic 1-minute data with embedded 'M' patterns."""
    rng = np.random.default_rng(42)
    index = pd.date_range(start='2023-01-01', periods=n_points, freq='min')
    price = np.zeros(n_points)

    price[0] = 100
    jumps = rng.normal(0, 0.1, n_points)
    price = 100 + np.cumsum(jumps)

    # Inject M-patterns with more volatility
    for i in range(250, n_points - 250, 500):
        price[i:i+30] += np.linspace(0, 5, 30) + rng.normal(0, 0.2, 30)
        price[i+30:i+60] -= np.linspace(0, 2.5, 30) + rng.normal(0, 0.2, 30)
        price[i+60:i+90] += np.linspace(0, 3, 30) + rng.normal(0, 0.2, 30)
        peak_high = np.max(price[i+60:i+90])

        price[i+90:i+150] -= np.linspace(0, 8, 60) + rng.normal(0, 0.3, 60)
        level_low = np.min(price[i+90:i+150])

        target_retracement = level_low + (peak_high - level_low) * 0.5
        current_price = price[i+149]
        price[i+150:i+180] += np.linspace(0, target_retracement - current_price, 30) + rng.normal(0, 0.2, 30)

        price[i+180:i+250] -= np.linspace(0, 7, 70) + rng.normal(0, 0.3, 70)

    df = pd.DataFrame({'Close': price}, index=index)
    df['Open'] = df['Close'].shift(1).fillna(method='bfill')
    high_noise = rng.uniform(0.05, 0.2, n_points)
    low_noise = rng.uniform(0.05, 0.2, n_points)
    df['High'] = df[['Open', 'Close']].max(axis=1) + high_noise
    df['Low'] = df[['Open', 'Close']].min(axis=1) - low_noise
    df['Volume'] = rng.integers(100, 1000, size=n_points)

    return df

# --- Pre-processing ---
def preprocess_data(data_1m: pd.DataFrame, prominence=1, width=5):
    """
    Simulates 15M timeframe analysis on 1M data to find M-pattern setups.
    """
    ohlc_15m = data_1m['Close'].resample('15min').ohlc()

    high_peaks_idx, _ = find_peaks(ohlc_15m['high'], prominence=prominence, width=width)
    low_peaks_idx, _ = find_peaks(-ohlc_15m['low'], prominence=prominence, width=width)

    ohlc_15m['swing_high'] = np.nan
    ohlc_15m.iloc[high_peaks_idx, ohlc_15m.columns.get_loc('swing_high')] = ohlc_15m['high'].iloc[high_peaks_idx]

    ohlc_15m['swing_low'] = np.nan
    ohlc_15m.iloc[low_peaks_idx, ohlc_15m.columns.get_loc('swing_low')] = ohlc_15m['low'].iloc[low_peaks_idx]

    ohlc_15m['last_swing_high'] = ohlc_15m['swing_high'].ffill()

    is_new_low = ohlc_15m['swing_low'].notna()
    is_level_drop = is_new_low & (ohlc_15m['swing_low'] < ohlc_15m['last_swing_high'])

    setup_df = ohlc_15m[is_level_drop].copy()

    setup_df['aoi_level'] = setup_df['swing_low'] + (setup_df['last_swing_high'] - setup_df['swing_low']) * 0.5
    setup_df['swing_low_for_tp'] = setup_df['swing_low']

    data_1m_sorted = data_1m.sort_index()
    setup_df_sorted = setup_df.sort_index()

    data_1m = pd.merge_asof(
        data_1m_sorted,
        setup_df_sorted[['aoi_level', 'swing_low_for_tp']],
        left_index=True,
        right_index=True,
        direction='backward'
    )

    return data_1m


class MeasuredMoveMwScalpStrategy(Strategy):
    min_rr = 4.0

    def init(self):
        self.state = TradeState.SEARCHING

    def next(self):
        current_aoi = self.data.aoi_level[-1]

        if self.state == TradeState.SEARCHING:
            if not np.isnan(current_aoi) and self.data.High[-1] >= current_aoi:
                self.state = TradeState.IN_AOI

        elif self.state == TradeState.IN_AOI:
            if self.data.Low[-1] < self.data.swing_low_for_tp[-1]:
                 self.state = TradeState.SEARCHING
                 return

            if self.data.Close[-1] < self.data.Open[-1]:
                if self.position:
                    return

                entry_price = self.data.Close[-1]
                reversal_high = self.data.High[-1]
                level_drop_low = self.data.swing_low_for_tp[-1]
                stop_loss = reversal_high

                if reversal_high <= level_drop_low or np.isnan(level_drop_low):
                    return

                target_price = reversal_high - (reversal_high - level_drop_low) * 0.5

                risk = abs(entry_price - stop_loss)
                reward = abs(entry_price - target_price)
                if risk == 0 or reward / risk < self.min_rr:
                    return

                size = (self.equity * RISK_PER_TRADE_PCT) / risk
                if size < 1:
                    return

                self.sell(size=int(size), sl=stop_loss, tp=target_price)
                self.state = TradeState.SEARCHING

if __name__ == '__main__':
    data = generate_m_pattern_data(n_points=5000)
    data = preprocess_data(data)

    if data.empty:
        raise ValueError("Data preprocessing resulted in an empty DataFrame.")

    bt = Backtest(data, MeasuredMoveMwScalpStrategy, cash=INITIAL_CASH, commission=COMMISSION_PCT)

    stats = bt.optimize(min_rr=np.arange(2, 6, 0.5).tolist(), maximize='Sharpe Ratio')
    print(stats)

    os.makedirs('results', exist_ok=True)

    if stats is not None and stats.get('# Trades', 0) > 0:
        output_stats = {
            'strategy_name': 'measured_move_mw_scalp',
            'return': stats.get('Return [%]', None),
            'sharpe': stats.get('Sharpe Ratio', None),
            'max_drawdown': stats.get('Max. Drawdown [%]', None),
            'win_rate': stats.get('Win Rate [%]', None),
            'total_trades': stats.get('# Trades', 0)
        }
        for key, value in output_stats.items():
            if isinstance(value, np.generic):
                output_stats[key] = value.item()
        with open('results/temp_result.json', 'w') as f:
            json.dump(output_stats, f, indent=2)
    else:
        with open('results/temp_result.json', 'w') as f:
            json.dump({
                'strategy_name': 'measured_move_mw_scalp',
                'return': None, 'sharpe': None, 'max_drawdown': None,
                'win_rate': None, 'total_trades': 0
            }, f, indent=2)

    print("Backtest stats saved to results/temp_result.json")

    if stats is not None and stats.get('# Trades', 0) > 0:
        bt.run(**stats['_strategy']._params)
        # The bt.plot() call is commented out due to a dependency issue between the old
        # `backtesting.py` library and a newer version of `bokeh` that causes a `ValueError`.
        # The core logic for backtesting and saving results is functional.
        # bt.plot(filename='results/measured_move_mw_scalp.html', open_browser=False)
        # print("Backtest plot saved to results/measured_move_mw_scalp.html")
    else:
        print("No trades were executed, skipping plot generation.")
