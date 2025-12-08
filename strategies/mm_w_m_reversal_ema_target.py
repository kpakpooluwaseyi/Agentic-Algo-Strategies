
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import json

# Define EMA function for backtesting.py
def EMA(series, period):
    return pd.Series(series).ewm(span=period, adjust=False).mean()

class MmWMReversalEmaTargetStrategy(Strategy):
    # Parameters for optimization
    w_lookback = 50
    m_lookback = 50
    ema_200_period = 200
    ema_800_period = 800

    def init(self):
        # Initialize indicators
        self.ema200 = self.I(EMA, self.data.Close, self.ema_200_period)
        self.ema800 = self.I(EMA, self.data.Close, self.ema_800_period)

        # State management for trades
        self.tp1_hit = False

    def is_hammer(self, o, h, l, c):
        """Checks for a Hammer candlestick pattern."""
        body = abs(c - o)
        if body == 0: return False
        lower_shadow = c - l if c > o else o - l
        upper_shadow = h - c if c > o else h - o
        return lower_shadow > 2 * body and upper_shadow < body * 0.7

    def is_inverted_hammer(self, o, h, l, c):
        """Checks for an Inverted Hammer candlestick pattern."""
        body = abs(c - o)
        if body == 0: return False
        lower_shadow = c - l if c > o else o - l
        upper_shadow = h - c if c > o else h - o
        return upper_shadow > 2 * body and lower_shadow < body * 0.7

    def next(self):
        # --- EXIT AND RISK MANAGEMENT LOGIC ---
        if self.position:
            # Trailing Stop after TP1
            if self.tp1_hit:
                if self.position.is_long:
                    new_sl = self.data.Low[-2]
                    if self.trade.sl < new_sl:
                        self.trade.sl = new_sl
                elif self.position.is_short:
                    new_sl = self.data.High[-2]
                    if self.trade.sl > new_sl:
                        self.trade.sl = new_sl

            # Take Profit and Rejection Logic
            if self.position.is_long:
                # Rejection Exit: Close if price touches 200 EMA but closes below it
                if not self.tp1_hit and self.data.High[-1] >= self.ema200[-1] and self.data.Close[-1] < self.ema200[-1]:
                    self.position.close()
                    return

                # Take Profit 1: 50% at 200 EMA
                if not self.tp1_hit and crossover(self.data.Close, self.ema200):
                    self.position.close(size=0.5)
                    self.tp1_hit = True
                    self.trade.sl = self.trade.entry_price # Move SL to Break-even

                # Take Profit 2: Rest at 800 EMA
                if crossover(self.data.Close, self.ema800):
                    self.position.close()

            elif self.position.is_short:
                # Rejection Exit: Close if price touches 200 EMA but closes above it
                if not self.tp1_hit and self.data.Low[-1] <= self.ema200[-1] and self.data.Close[-1] > self.ema200[-1]:
                    self.position.close()
                    return

                # Take Profit 1: 50% at 200 EMA
                if not self.tp1_hit and crossover(self.ema200, self.data.Close):
                    self.position.close(size=0.5)
                    self.tp1_hit = True
                    self.trade.sl = self.trade.entry_price # Move SL to Break-even

                # Take Profit 2: Rest at 800 EMA
                if crossover(self.ema800, self.data.Close):
                    self.position.close()

        # --- ENTRY LOGIC ---
        if not self.position:
            # LONG ENTRY: W-Formation (Double Bottom)
            if len(self.data.Close) > self.w_lookback:
                # Find the most recent major low (potential T2)
                recent_low_idx = np.argmin(self.data.Low[-self.w_lookback:])
                # Check if the current bar is this recent low
                if recent_low_idx == self.w_lookback - 1:
                    current_low = self.data.Low[-1]

                    # Find a peak before this low
                    peak_idx_local = np.argmax(self.data.High[-self.w_lookback:-1])

                    # Find T1 before the peak
                    window_for_t1 = self.data.Low[-self.w_lookback : -self.w_lookback + peak_idx_local]
                    if len(window_for_t1) > 3:
                        t1_low = np.min(window_for_t1)
                        # Condition: T2 is a lower low than T1
                        if current_low < t1_low:
                            # Condition: A Hammer appears at T2
                            if self.is_hammer(self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1]):
                                self.buy(sl=current_low)
                                self.tp1_hit = False
                                return

            # SHORT ENTRY: M-Formation (Double Top)
            if len(self.data.Close) > self.m_lookback:
                # Find the most recent major high (potential P2)
                recent_high_idx = np.argmax(self.data.High[-self.m_lookback:])
                # Check if the current bar is this recent high
                if recent_high_idx == self.m_lookback - 1:
                    current_high = self.data.High[-1]

                    # Find a trough before this high
                    trough_idx_local = np.argmin(self.data.Low[-self.m_lookback:-1])

                    # Find P1 before the trough
                    window_for_p1 = self.data.High[-self.m_lookback : -self.m_lookback + trough_idx_local]
                    if len(window_for_p1) > 3:
                        p1_high = np.max(window_for_p1)
                        # Condition: P2 is a higher high than P1
                        if current_high > p1_high:
                            # Condition: An Inverted Hammer appears at P2
                            if self.is_inverted_hammer(self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1]):
                                self.sell(sl=current_high)
                                self.tp1_hit = False
                                return

if __name__ == '__main__':
    try:
        from backtesting.test import GOOG
        data = GOOG.copy()
        data = data.iloc[-1500:]
    except ImportError:
        from backtesting.test import EURUSD
        data = EURUSD.copy()
        data = data.iloc[-1500:]

    # Run backtest with finalize_trades=True
    bt = Backtest(data, MmWMReversalEmaTargetStrategy, cash=10000, commission=.002, finalize_trades=True)

    # Optimize
    stats = bt.optimize(
        w_lookback=range(20, 80, 10),
        m_lookback=range(20, 80, 10),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.w_lookback >= 20 and p.m_lookback >= 20
    )

    print("Best optimization results:")
    print(stats)

    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        stats_dict = {
            'strategy_name': 'mm_w_m_reversal_ema_target',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }
        json.dump(stats_dict, f, indent=2)

    # Generate plot
    bt.plot()
