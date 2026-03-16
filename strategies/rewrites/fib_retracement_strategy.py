"""
Strategy 4: Fibonacci Retracement Entry
========================================
Simple Fib retracement strategy.
Enter on 61.8% retracement in trending markets.

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json


def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    return pd.Series(tr).rolling(period).mean().values


class FibRetracementStrategy(Strategy):
    """
    Entry: 61.8% Fib retracement in trend
    Exit: 0% Fib (swing high) or stop at 100%
    """
    
    swing_lookback = 20
    fib_level = 0.618
    fib_tolerance = 0.05  # 5% tolerance
    risk_per_trade = 0.02
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Swing detection
        self.swing_high = self.I(lambda: high.rolling(self.swing_lookback).max().values)
        self.swing_low = self.I(lambda: low.rolling(self.swing_lookback).min().values)
        
        # Trend filter
        ema_50 = ta.ema(close, length=50)
        ema_200 = ta.ema(close, length=200)
        self.ema_50 = self.I(lambda: ema_50.values)
        self.ema_200 = self.I(lambda: ema_200.values)
        
        self.atr = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, 14)
    
    def next(self):
        if self.position or len(self.data) < 210:
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        swing_h = self.swing_high[-1]
        swing_l = self.swing_low[-1]
        
        if np.isnan(swing_h) or np.isnan(swing_l) or swing_h <= swing_l:
            return
        
        move_size = swing_h - swing_l
        fib_618 = swing_h - (move_size * self.fib_level)
        tolerance = move_size * self.fib_tolerance
        
        # Trend
        uptrend = self.ema_50[-1] > self.ema_200[-1] if not np.isnan(self.ema_200[-1]) else False
        downtrend = self.ema_50[-1] < self.ema_200[-1] if not np.isnan(self.ema_200[-1]) else False
        
        # Long: Price at 61.8% retracement in uptrend
        if uptrend and abs(price - fib_618) <= tolerance:
            bullish_candle = self.data.Close[-1] > self.data.Open[-1]
            if bullish_candle:
                sl = swing_l - (atr * 0.5)
                tp = swing_h + (move_size * 0.5)  # 150% extension
                risk = price - sl
                if risk > 0:
                    shares = max(1, min(int((self.equity * self.risk_per_trade) / risk), 
                                       int(self.equity * 0.5 / price)))
                    self.buy(size=shares, sl=sl, tp=tp)
        
        # Short: Price at 38.2% (inverse) in downtrend
        fib_382 = swing_l + (move_size * (1 - self.fib_level))
        if downtrend and abs(price - fib_382) <= tolerance:
            bearish_candle = self.data.Close[-1] < self.data.Open[-1]
            if bearish_candle:
                sl = swing_h + (atr * 0.5)
                tp = swing_l - (move_size * 0.5)
                risk = sl - price
                if risk > 0:
                    shares = max(1, min(int((self.equity * self.risk_per_trade) / risk), 
                                       int(self.equity * 0.5 / price)))
                    self.sell(size=shares, sl=sl, tp=tp)


if __name__ == '__main__':
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_4h_200weeks.csv')
    
    print("="*60)
    print("FIB RETRACEMENT STRATEGY - STRATEGY 4")
    print("="*60)
    
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, FibRetracementStrategy, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    ret = stats['Return [%]'] if pd.notna(stats['Return [%]']) else 0
    sharpe = stats['Sharpe Ratio'] if pd.notna(stats['Sharpe Ratio']) else 0
    
    print(f"Return: {ret:.2f}%, Sharpe: {sharpe:.3f}, Trades: {stats['# Trades']}")
    
    result = {'strategy': 'fib_retracement', 'return': float(ret), 'sharpe': float(sharpe), 'trades': int(stats['# Trades'])}
    os.makedirs('results', exist_ok=True)
    with open('results/fib_retracement_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("✅ PROFITABLE" if ret > 0 else "❌ NEEDS WORK")
