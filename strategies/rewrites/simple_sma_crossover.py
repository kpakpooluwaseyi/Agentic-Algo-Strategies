"""
Simple SMA Crossover Strategy
==============================
The most basic trend-following approach.
No complex indicators - just price vs moving average.

If this doesn't work, the data/market regime is fundamentally 
hostile to systematic trend-following.

Entry:
- LONG: Close > SMA50 and SMA50 > SMA200
- SHORT: Close < SMA50 and SMA50 < SMA200

Exit:
- LONG: Close < SMA50
- SHORT: Close > SMA50

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json


class SimpleSMACrossover(Strategy):
    """Ultra-simple SMA crossover."""
    
    sma_fast = 50
    sma_slow = 200
    risk_per_trade = 0.02
    
    def init(self):
        close = pd.Series(self.data.Close)
        
        sma_fast = ta.sma(close, length=self.sma_fast)
        sma_slow = ta.sma(close, length=self.sma_slow)
        
        self.sma_fast = self.I(lambda: sma_fast.values)
        self.sma_slow = self.I(lambda: sma_slow.values)
    
    def next(self):
        if len(self.data) < 210:
            return
        
        price = self.data.Close[-1]
        sma_f = self.sma_fast[-1]
        sma_s = self.sma_slow[-1]
        
        if np.isnan(sma_f) or np.isnan(sma_s):
            return
        
        # Define trend
        uptrend = sma_f > sma_s
        downtrend = sma_f < sma_s
        
        # Current position vs trend
        in_correct_position = False
        
        if self.position:
            if self.position.is_long and not uptrend:
                self.position.close()
            elif self.position.is_short and not downtrend:
                self.position.close()
            else:
                in_correct_position = True
        
        # Entry
        if not self.position and not in_correct_position:
            # Position sizing: 2% of equity
            max_shares = int(self.equity * self.risk_per_trade / price)
            if max_shares < 1:
                max_shares = 1
            
            # LONG
            if price > sma_f and uptrend:
                self.buy(size=max_shares)
            
            # SHORT
            elif price < sma_f and downtrend:
                self.sell(size=max_shares)


def run_test():
    datasets = [
        ("BTC 1H (2021-2025)", "data/crypto/BTC-USDT_1h_200weeks.csv"),
        ("BTC 4H (2021-2025)", "data/crypto/BTC-USDT_4h_200weeks.csv"),
    ]
    
    print("="*60)
    print("SIMPLE SMA CROSSOVER TEST")
    print("="*60)
    
    for name, path in datasets:
        if not os.path.exists(path):
            continue
        
        try:
            data = pd.read_csv(path, parse_dates=[0], index_col=0)
            data.columns = [c.strip().capitalize() for c in data.columns]
            
            bt = Backtest(data, SimpleSMACrossover, cash=1_000_000, 
                         commission=0.002, trade_on_close=True)
            stats = bt.run()
            
            ret = stats['Return [%]'] if pd.notna(stats['Return [%]']) else 0
            sharpe = stats['Sharpe Ratio'] if pd.notna(stats['Sharpe Ratio']) else 0
            trades = stats['# Trades']
            wr = stats['Win Rate [%]'] if pd.notna(stats['Win Rate [%]']) else 0
            
            print(f"\n{name}:")
            print(f"  Return:    {ret:.1f}%")
            print(f"  Sharpe:    {sharpe:.2f}")
            print(f"  Trades:    {trades}")
            print(f"  Win Rate:  {wr:.1f}%")
            
        except Exception as e:
            print(f"  ❌ {name}: {str(e)[:50]}")


if __name__ == '__main__':
    run_test()
