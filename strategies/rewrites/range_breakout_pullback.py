"""
Strategy 9: Range Breakout Pullback
====================================
Breakout from consolidation range with pullback entry.

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json


class RangeBreakoutPullback(Strategy):
    """
    Detect consolidation range, enter on pullback after breakout.
    """
    
    range_period = 20
    atr_mult = 0.5  # Range if ATR < 0.5 * avg ATR
    risk_per_trade = 0.02
    
    def init(self):
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        
        self.range_high = self.I(lambda: high.rolling(self.range_period).max().values)
        self.range_low = self.I(lambda: low.rolling(self.range_period).min().values)
        
        atr = ta.atr(high, low, close, length=14)
        self.atr = self.I(lambda: atr.values)
        self.atr_avg = self.I(lambda: atr.rolling(50).mean().values)
        
        self.broke_up = False
        self.broke_down = False
        self.breakout_level = None
    
    def next(self):
        if len(self.data) < 60:
            return
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr):
            return
        
        range_h = self.range_high[-2]  # Use prior bar's range
        range_l = self.range_low[-2]
        
        # Manage position
        if self.position:
            return
        
        # Detect breakout
        if high > range_h and not self.broke_up:
            self.broke_up = True
            self.breakout_level = range_h
            self.broke_down = False
        elif low < range_l and not self.broke_down:
            self.broke_down = True
            self.breakout_level = range_l
            self.broke_up = False
        
        shares = max(1, int(self.equity * self.risk_per_trade / price))
        
        # Entry on pullback to breakout level
        if self.broke_up and self.breakout_level:
            if low <= self.breakout_level and price > self.breakout_level:
                sl = range_l
                tp = price + (range_h - range_l)
                if sl < price:
                    self.buy(size=shares, sl=sl, tp=tp)
                    self.broke_up = False
        
        elif self.broke_down and self.breakout_level:
            if high >= self.breakout_level and price < self.breakout_level:
                sl = range_h
                tp = price - (range_h - range_l)
                if sl > price:
                    self.sell(size=shares, sl=sl, tp=tp)
                    self.broke_down = False


if __name__ == '__main__':
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_4h_200weeks.csv')
    
    print("="*60)
    print("RANGE BREAKOUT PULLBACK - STRATEGY 9")
    print("="*60)
    
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, RangeBreakoutPullback, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    ret = stats['Return [%]'] if pd.notna(stats['Return [%]']) else 0
    sharpe = stats['Sharpe Ratio'] if pd.notna(stats['Sharpe Ratio']) else 0
    
    print(f"Return: {ret:.2f}%, Sharpe: {sharpe:.3f}, Trades: {stats['# Trades']}")
    print("✅ PROFITABLE" if ret > 0 else "❌ NEEDS WORK")
    
    os.makedirs('results', exist_ok=True)
    with open('results/range_breakout_result.json', 'w') as f:
        json.dump({'strategy': 'range_breakout', 'return': float(ret), 'sharpe': float(sharpe), 'trades': int(stats['# Trades'])}, f)
