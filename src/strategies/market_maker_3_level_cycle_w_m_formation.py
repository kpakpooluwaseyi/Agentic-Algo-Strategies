from backtesting import Backtest, Strategy
from backtesting.test import GOOG
import pandas as pd
import numpy as np
import json
import os

def pivot_indicator(series, lookback: int, is_high: bool):
    """
    Identifies pivot points in a series.
    A pivot high is a value that is greater than all the values in the lookback window.
    A pivot low is a value that is smaller than all the values in the lookback window.
    """
    series = pd.Series(series)

    if is_high:
        return series.rolling(lookback * 2 + 1, center=True, min_periods=1).max() == series
    else:
        return series.rolling(lookback * 2 + 1, center=True, min_periods=1).min() == series

class MarketMaker3LevelCycleWMFormationStrategy(Strategy):
    """
    This strategy implements a simplified version of the Market Maker W and M formation trading.
    It identifies W-formations (double bottoms with a higher low) for long entries and
    M-formations (double tops with a lower high) for short entries.

    The strategy simplifies or omits ambiguous concepts from the request like "MM Cycle Levels"
    and "MM Candles" in favor of a pure, backtestable algorithm based on price action patterns.
    """
    pivot_lookback = 15
    risk_reward_ratio = 2.0
    history_len = 100

    def init(self):
        self.is_high_pivot = self.I(pivot_indicator, self.data.High, self.pivot_lookback, True)
        self.is_low_pivot = self.I(pivot_indicator, self.data.Low, self.pivot_lookback, False)

    def next(self):
        if self.position and self.data.index[-1].weekday() == 4: # Friday
            self.position.close()
            return

        if self.position:
            return

        if len(self.data.Close) < self.history_len:
            return

        high_pivots_bool = self.is_high_pivot[-self.history_len:]
        low_pivots_bool = self.is_low_pivot[-self.history_len:]

        high_pivot_indices = np.where(high_pivots_bool)[0]
        low_pivot_indices = np.where(low_pivots_bool)[0]

        # M-Formation (Short Entry)
        if len(high_pivot_indices) >= 2 and len(low_pivot_indices) >= 1:
            H2_local_idx, H1_local_idx = high_pivot_indices[-1], high_pivot_indices[-2]
            troughs_between = low_pivot_indices[(low_pivot_indices > H1_local_idx) & (low_pivot_indices < H2_local_idx)]

            if len(troughs_between) > 0:
                trough_local_idx = troughs_between[-1]
                H1_val = self.data.High[-self.history_len:][H1_local_idx]
                H2_val = self.data.High[-self.history_len:][H2_local_idx]
                trough_val = self.data.Low[-self.history_len:][trough_local_idx]

                is_lower_high = H2_val < H1_val
                is_recent = H2_local_idx >= self.history_len - self.pivot_lookback - 2

                if is_lower_high and is_recent and self.data.Close[-1] < trough_val:
                    sl = max(H1_val, H2_val)
                    tp = self.data.Close[-1] - (sl - self.data.Close[-1]) * self.risk_reward_ratio
                    if tp > 0:
                        self.sell(sl=sl, tp=tp)
                    return

        # W-Formation (Long Entry)
        if len(low_pivot_indices) >= 2 and len(high_pivot_indices) >= 1:
            L2_local_idx, L1_local_idx = low_pivot_indices[-1], low_pivot_indices[-2]
            peaks_between = high_pivot_indices[(high_pivot_indices > L1_local_idx) & (high_pivot_indices < L2_local_idx)]

            if len(peaks_between) > 0:
                peak_local_idx = peaks_between[-1]
                L1_val = self.data.Low[-self.history_len:][L1_local_idx]
                L2_val = self.data.Low[-self.history_len:][L2_local_idx]
                peak_val = self.data.High[-self.history_len:][peak_local_idx]

                is_higher_low = L2_val > L1_val
                is_recent = L2_local_idx >= self.history_len - self.pivot_lookback - 2

                if is_higher_low and is_recent and self.data.Close[-1] > peak_val:
                    sl = min(L1_val, L2_val)
                    tp = self.data.Close[-1] + (self.data.Close[-1] - sl) * self.risk_reward_ratio
                    self.buy(sl=sl, tp=tp)
                    return

if __name__ == '__main__':
    data = GOOG
    bt = Backtest(data, MarketMaker3LevelCycleWMFormationStrategy, cash=10000, commission=.002)
    stats = bt.optimize(
        pivot_lookback=range(5, 25, 5),
        risk_reward_ratio=[1.0, 1.5, 2.0, 2.5],
        maximize='Sharpe Ratio'
    )

    print("Best stats:", stats)

    os.makedirs('results', exist_ok=True)

    # Extract serializable parameters, casting numpy types to native Python types
    params = {
        'pivot_lookback': int(stats._strategy.pivot_lookback),
        'risk_reward_ratio': float(stats._strategy.risk_reward_ratio)
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'market_maker_3_level_cycle_w_m_formation',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades']),
            'parameters': params
        }, f, indent=2)

    bt.plot()
