"""
Strategy 8: Triple Reversal Pattern
=====================================
Three consecutive opposing candles signal reversal.

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json


class TripleReversalPattern(Strategy):
    """
    Long: 3 consecutive red candles + bullish engulfing
    Short: 3 consecutive green candles + bearish engulfing
    """
    
    risk_per_trade = 0.02
    
    def init(self):
        close = pd.Series(self.data.Close)
        ema_50 = ta.ema(close, length=50)
        self.ema_50 = self.I(lambda: ema_50.values)
    
    def next(self):
        if self.position or len(self.data) < 10:
            return
        
        price = self.data.Close[-1]
        
        # Check for 3 consecutive reds followed by bullish
        if len(self.data) >= 5:
            c1_red = self.data.Close[-4] < self.data.Open[-4]
            c2_red = self.data.Close[-3] < self.data.Open[-3]
            c3_red = self.data.Close[-2] < self.data.Open[-2]
            c4_bullish = self.data.Close[-1] > self.data.Open[-1]
            engulf_bull = self.data.Close[-1] > self.data.Open[-2]
            
            c1_green = self.data.Close[-4] > self.data.Open[-4]
            c2_green = self.data.Close[-3] > self.data.Open[-3]
            c3_green = self.data.Close[-2] > self.data.Open[-2]
            c4_bearish = self.data.Close[-1] < self.data.Open[-1]
            engulf_bear = self.data.Close[-1] < self.data.Open[-2]
            
            shares = max(1, int(self.equity * self.risk_per_trade / price))
            
            # Long: 3 reds + bullish engulfing
            if c1_red and c2_red and c3_red and c4_bullish and engulf_bull:
                sl = self.data.Low[-2] * 0.99
                tp = price + (price - sl) * 2
                self.buy(size=shares, sl=sl, tp=tp)
            
            # Short: 3 greens + bearish engulfing
            elif c1_green and c2_green and c3_green and c4_bearish and engulf_bear:
                sl = self.data.High[-2] * 1.01
                tp = price - (sl - price) * 2
                self.sell(size=shares, sl=sl, tp=tp)


if __name__ == '__main__':
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_4h_200weeks.csv')
    
    print("="*60)
    print("TRIPLE REVERSAL - STRATEGY 8")
    print("="*60)
    
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, TripleReversalPattern, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    ret = stats['Return [%]'] if pd.notna(stats['Return [%]']) else 0
    sharpe = stats['Sharpe Ratio'] if pd.notna(stats['Sharpe Ratio']) else 0
    
    print(f"Return: {ret:.2f}%, Sharpe: {sharpe:.3f}, Trades: {stats['# Trades']}")
    print("✅ PROFITABLE" if ret > 0 else "❌ NEEDS WORK")
    
    os.makedirs('results', exist_ok=True)
    with open('results/triple_reversal_result.json', 'w') as f:
        json.dump({'strategy': 'triple_reversal', 'return': float(ret), 'sharpe': float(sharpe), 'trades': int(stats['# Trades'])}, f)
