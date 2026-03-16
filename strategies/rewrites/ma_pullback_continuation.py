"""
Strategy 7: MA Pullback Continuation
=====================================
Enter on pullback to moving average in strong trend.

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json


class MAPullbackContinuation(Strategy):
    """
    Long: Uptrend + pullback to 20 EMA + bounce
    Short: Downtrend + pullback to 20 EMA + rejection
    """
    
    ema_fast = 20
    ema_slow = 50
    risk_per_trade = 0.02
    
    def init(self):
        close = pd.Series(self.data.Close)
        
        ema_20 = ta.ema(close, length=self.ema_fast)
        ema_50 = ta.ema(close, length=self.ema_slow)
        
        self.ema_20 = self.I(lambda: ema_20.values)
        self.ema_50 = self.I(lambda: ema_50.values)
    
    def next(self):
        if self.position or len(self.data) < 60:
            return
        
        price = self.data.Close[-1]
        low = self.data.Low[-1]
        high = self.data.High[-1]
        
        if np.isnan(self.ema_20[-1]) or np.isnan(self.ema_50[-1]):
            return
        
        uptrend = self.ema_20[-1] > self.ema_50[-1]
        downtrend = self.ema_20[-1] < self.ema_50[-1]
        
        # Pullback to EMA20: low touched EMA but closed above
        touched_ema_from_above = low <= self.ema_20[-1] and price > self.ema_20[-1]
        touched_ema_from_below = high >= self.ema_20[-1] and price < self.ema_20[-1]
        
        shares = max(1, int(self.equity * self.risk_per_trade / price))
        
        # Long: uptrend + pullback to EMA20
        if uptrend and touched_ema_from_above:
            bullish = price > self.data.Open[-1]
            if bullish:
                sl = low * 0.98
                tp = price + (price - sl) * 2
                self.buy(size=shares, sl=sl, tp=tp)
        
        # Short: downtrend + rally to EMA20
        elif downtrend and touched_ema_from_below:
            bearish = price < self.data.Open[-1]
            if bearish:
                sl = high * 1.02
                tp = price - (sl - price) * 2
                self.sell(size=shares, sl=sl, tp=tp)


if __name__ == '__main__':
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_4h_200weeks.csv')
    
    print("="*60)
    print("MA PULLBACK CONTINUATION - STRATEGY 7")
    print("="*60)
    
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, MAPullbackContinuation, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    ret = stats['Return [%]'] if pd.notna(stats['Return [%]']) else 0
    sharpe = stats['Sharpe Ratio'] if pd.notna(stats['Sharpe Ratio']) else 0
    
    print(f"Return: {ret:.2f}%, Sharpe: {sharpe:.3f}, Trades: {stats['# Trades']}")
    print("✅ PROFITABLE" if ret > 0 else "❌ NEEDS WORK")
    
    os.makedirs('results', exist_ok=True)
    with open('results/ma_pullback_result.json', 'w') as f:
        json.dump({'strategy': 'ma_pullback', 'return': float(ret), 'sharpe': float(sharpe), 'trades': int(stats['# Trades'])}, f)
