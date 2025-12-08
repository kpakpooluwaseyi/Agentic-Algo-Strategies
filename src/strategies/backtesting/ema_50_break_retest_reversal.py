from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd

def EMA(series, period):
    """Calculates the Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, adjust=False).mean()

class Ema50BreakRetestReversalStrategy(Strategy):
    """
    Strategy based on the EMA 50 break, retest, and reversal pattern.
    """
    ema_fast_period = 50
    ema_slow_period = 800
    time_exit_bars = 3

    def init(self):
        """Initialize the strategy indicators and state variables."""
        self.ema50 = self.I(EMA, self.data.Close, self.ema_fast_period)
        self.ema800 = self.I(EMA, self.data.Close, self.ema_slow_period)

        # State for short trade pattern
        self.short_break_below_ema50 = False
        self.short_retest_high = 0

        # State for long trade pattern
        self.long_break_above_ema50 = False
        self.long_retest_low = float('inf')

    def next(self):
        """Define the strategy logic for the next tick."""
        # --- Handle Position Exits ---
        if self.trades:
            trade = self.trades[0]
            # Time-based exit
            if self.data.index[-1] - trade.entry_time >= pd.Timedelta(hours=self.time_exit_bars):
                self.position.close()

        # --- Handle Position Entries ---

        # --- SHORT TRADE LOGIC ---
        # 1. Detect the initial break below the 50 EMA
        if not self.short_break_below_ema50 and crossover(self.ema50, self.data.Close):
            self.short_break_below_ema50 = True
            self.short_retest_high = 0
            self.long_break_above_ema50 = False  # Invalidate any ongoing long pattern
            self.long_retest_low = float('inf')

        # 2. If a break has occurred, monitor for a retest
        if self.short_break_below_ema50:
            # Track the highest price during the retest phase
            self.short_retest_high = max(self.short_retest_high, self.data.High[-1])

            # 3. Entry Confirmation: Price closes below EMA50 again
            if crossover(self.ema50, self.data.Close):
                if self.short_retest_high > 0 and not self.position:
                    # Set Stop Loss slightly above the retest high
                    sl = self.short_retest_high * 1.001
                    # Set Take Profit at the 800 EMA
                    tp = self.ema800[-1]
                    if tp < self.data.Close[-1] < sl:
                        self.sell(sl=sl, tp=tp)

                # Reset the pattern state after the trade
                self.short_break_below_ema50 = False
                self.short_retest_high = 0

        # --- LONG TRADE LOGIC (REVERSE OF SHORT) ---
        # 1. Detect the initial break above the 50 EMA
        if not self.long_break_above_ema50 and crossover(self.data.Close, self.ema50):
            self.long_break_above_ema50 = True
            self.long_retest_low = float('inf')
            self.short_break_below_ema50 = False  # Invalidate any ongoing short pattern
            self.short_retest_high = 0

        # 2. If a break has occurred, monitor for a retest
        if self.long_break_above_ema50:
            # Track the lowest price during the retest phase
            self.long_retest_low = min(self.long_retest_low, self.data.Low[-1])

            # 3. Entry Confirmation: Price closes above EMA50 again
            if crossover(self.data.Close, self.ema50):
                if self.long_retest_low < float('inf') and not self.position:
                    # Set Stop Loss slightly below the retest low
                    sl = self.long_retest_low * 0.999
                    # Set Take Profit at the 800 EMA
                    tp = self.ema800[-1]
                    if sl < self.data.Close[-1] < tp:
                        self.buy(sl=sl, tp=tp)

                # Reset the pattern state after the trade
                self.long_break_above_ema50 = False
                self.long_retest_low = float('inf')

if __name__ == '__main__':
    from backtesting import Backtest
    from backtesting.test import GOOG
    import json
    import os

    # Load data
    data = GOOG.copy()
    data.columns = [column.capitalize() for column in data.columns]


    # Run backtest
    bt = Backtest(data, Ema50BreakRetestReversalStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(
        ema_fast_period=range(40, 60, 5),
        ema_slow_period=range(700, 900, 50),
        time_exit_bars=range(3, 6, 1),
        maximize='Sharpe Ratio'
    )

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'ema_50_break_retest_reversal',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    # Generate plot
    bt.plot()
