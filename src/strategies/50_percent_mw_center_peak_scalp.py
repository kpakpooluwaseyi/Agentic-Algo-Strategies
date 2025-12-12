
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
from enum import Enum

# --- Helper function for EMA ---
def EMA(series, period):
    """Calculates the Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, adjust=False).mean()

# --- State Machines for Long and Short Setups ---
class ShortState(Enum):
    SEARCH_A = 0
    SEARCH_B = 1
    SEARCH_C = 2

class LongState(Enum):
    SEARCH_A = 0
    SEARCH_B = 1
    SEARCH_C = 2

class FiftyPercentMwCenterPeakScalpStrategy(Strategy):
    """
    Implements the 50% M/W Center Peak Scalp strategy.
    """
    major_lookback = 50
    minor_lookback = 3
    fib_tolerance = 0.05
    sl_buffer_pct = 0.02
    ema_period = 100
    reversal_pct = 0.03

    def init(self):
        self.ema = self.I(EMA, self.data.Close, self.ema_period)
        self.short_state = ShortState.SEARCH_A
        self.short_A_price, self.short_A_idx, self.short_B_price = None, None, None
        self.short_aoi_lower, self.short_aoi_upper = None, None
        self.long_state = LongState.SEARCH_A
        self.long_A_price, self.long_A_idx, self.long_B_price = None, None, None
        self.long_aoi_lower, self.long_aoi_upper = None, None

    def next(self):
        if len(self.data.Close) < self.major_lookback + 5:
            return
        if self.position:
            return
        self.run_short_logic()
        self.run_long_logic()

    def run_short_logic(self):
        if self.data.Close[-1] > self.ema[-1]:
            self.short_state = ShortState.SEARCH_A
            return

        current_idx = len(self.data.Close) - 1

        if self.short_state == ShortState.SEARCH_A:
            swing_high = np.max(self.data.High[-self.major_lookback:])
            if self.data.High[-1] < swing_high * (1 - self.reversal_pct):
                self.short_A_price = swing_high
                self.short_A_idx = current_idx - np.argmax(self.data.High[-self.major_lookback:])
                self.short_state = ShortState.SEARCH_B

        elif self.short_state == ShortState.SEARCH_B:
            if self.data.High[-1] > self.short_A_price:
                self.short_state = ShortState.SEARCH_A
                return

            if current_idx > self.short_A_idx:
                window = self.data.Low[self.short_A_idx:current_idx+1]
                swing_low_b = np.min(window)
                if self.data.Low[-1] > swing_low_b * (1 + self.reversal_pct * 0.5):
                    self.short_B_price = swing_low_b
                    level_drop = self.short_A_price - self.short_B_price
                    if level_drop > 0:
                        self.short_aoi_lower = self.short_B_price + level_drop * (0.5 - self.fib_tolerance)
                        self.short_aoi_upper = self.short_B_price + level_drop * (0.5 + self.fib_tolerance)
                        self.short_state = ShortState.SEARCH_C

        elif self.short_state == ShortState.SEARCH_C:
            if self.data.High[-1] > self.short_A_price or self.data.Low[-1] < self.short_B_price:
                self.short_state = ShortState.SEARCH_A
                return

            if self.short_aoi_lower <= self.data.High[-1] <= self.short_aoi_upper:
                is_confirmed = all(self.data.Close[-i] < self.data.Open[-i] for i in range(1, self.minor_lookback + 1))
                if is_confirmed:
                    sl = self.data.High[-1] * (1 + self.sl_buffer_pct)
                    tp = self.data.Close[-1] - (self.data.High[-1] - self.short_B_price) * 0.5
                    if tp < self.data.Close[-1]:
                        self.sell(sl=sl, tp=tp)
                    self.short_state = ShortState.SEARCH_A

    def run_long_logic(self):
        if self.data.Close[-1] < self.ema[-1]:
            self.long_state = LongState.SEARCH_A
            return

        current_idx = len(self.data.Close) - 1

        if self.long_state == LongState.SEARCH_A:
            swing_low = np.min(self.data.Low[-self.major_lookback:])
            if self.data.Low[-1] > swing_low * (1 + self.reversal_pct):
                self.long_A_price = swing_low
                self.long_A_idx = current_idx - np.argmin(self.data.Low[-self.major_lookback:])
                self.long_state = LongState.SEARCH_B

        elif self.long_state == LongState.SEARCH_B:
            if self.data.Low[-1] < self.long_A_price:
                self.long_state = LongState.SEARCH_A
                return

            if current_idx > self.long_A_idx:
                window = self.data.High[self.long_A_idx:current_idx+1]
                swing_high_b = np.max(window)
                if self.data.High[-1] < swing_high_b * (1 - self.reversal_pct * 0.5):
                    self.long_B_price = swing_high_b
                    level_rise = self.long_B_price - self.long_A_price
                    if level_rise > 0:
                        self.long_aoi_lower = self.long_B_price - level_rise * (0.5 + self.fib_tolerance)
                        self.long_aoi_upper = self.long_B_price - level_rise * (0.5 - self.fib_tolerance)
                        self.long_state = LongState.SEARCH_C

        elif self.long_state == LongState.SEARCH_C:
            if self.data.Low[-1] < self.long_A_price or self.data.High[-1] > self.long_B_price:
                self.long_state = LongState.SEARCH_A
                return

            if self.long_aoi_lower <= self.data.Low[-1] <= self.long_aoi_upper:
                is_confirmed = all(self.data.Close[-i] > self.data.Open[-i] for i in range(1, self.minor_lookback + 1))
                if is_confirmed:
                    sl = self.data.Low[-1] * (1 - self.sl_buffer_pct)
                    tp = self.data.Close[-1] + (self.long_B_price - self.data.Low[-1]) * 0.5
                    if tp > self.data.Close[-1]:
                        self.buy(sl=sl, tp=tp)
                    self.long_state = LongState.SEARCH_A

if __name__ == '__main__':
    from backtesting.test import GOOG
    data = GOOG.copy()
    bt = Backtest(data, FiftyPercentMwCenterPeakScalpStrategy, cash=100_000, commission=.002)

    # Run optimization
    best_stats = bt.optimize(
        major_lookback=range(40, 101, 20),
        minor_lookback=range(2, 5), # 2, 3, or 4 candle confirmation
        ema_period=range(50, 151, 50),
        reversal_pct=[i/100 for i in range(2, 5)],
        fib_tolerance=[i/100 for i in range(1, 4)],
        sl_buffer_pct=[i/1000 for i in range(10, 31, 10)],
        maximize='Sharpe Ratio',
        constraint=lambda p: p.major_lookback > p.minor_lookback
    )

    print("Best stats found from optimization:")
    print(best_stats)

    # --- Rerun with best parameters for plotting and final results ---
    # Note: Optimization returns a Series, not a dict. Best strategy params are in `_strategy`.
    best_params = best_stats._strategy._params
    print("\nRunning final backtest with best parameters:", best_params)
    bt = Backtest(data, FiftyPercentMwCenterPeakScalpStrategy, cash=100_000, commission=.002)
    final_stats = bt.run(**best_params)

    print("\nFinal stats with best parameters:")
    print(final_stats)

    # --- Save final results ---
    import os
    os.makedirs('results', exist_ok=True)

    total_trades = final_stats['# Trades']
    win_rate = final_stats.get('Win Rate [%]', 0)
    sharpe = final_stats.get('Sharpe Ratio', 0)

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': '50_percent_mw_center_peak_scalp',
            'return': final_stats.get('Return [%]', 0),
            'sharpe': sharpe,
            'max_drawdown': final_stats.get('Max. Drawdown [%]', 0),
            'win_rate': win_rate,
            'total_trades': total_trades,
        }, f, indent=2)

    print("\nFinal results saved to results/temp_result.json")

    # --- Plot the final run ---
    bt.plot()
