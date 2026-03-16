"""
Strategy 10: Wave Correction Entry
===================================
Enter on ABC correction completion (simplified wave analysis).

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json


class WaveCorrectionEntry(Strategy):
    """
    Simplified wave correction strategy.
    Look for 3-wave pullback in trend, enter on completion.
    """
    
    swing_lookback = 10
    risk_per_trade = 0.02
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Trend determination
        ema_50 = ta.ema(close, length=50)
        ema_200 = ta.ema(close, length=200)
        self.ema_50 = self.I(lambda: ema_50.values)
        self.ema_200 = self.I(lambda: ema_200.values)
        
        # RSI for exhaustion
        rsi = ta.rsi(close, length=14)
        self.rsi = self.I(lambda: rsi.values)
        
        # Swing levels
        self.swing_high = self.I(lambda: high.rolling(self.swing_lookback * 2 + 1, center=True).max().values)
        self.swing_low = self.I(lambda: low.rolling(self.swing_lookback * 2 + 1, center=True).min().values)
    
    def next(self):
        if self.position or len(self.data) < 210:
            return
        
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        
        if np.isnan(self.ema_50[-1]) or np.isnan(self.ema_200[-1]) or np.isnan(rsi):
            return
        
        uptrend = self.ema_50[-1] > self.ema_200[-1]
        downtrend = self.ema_50[-1] < self.ema_200[-1]
        
        shares = max(1, int(self.equity * self.risk_per_trade / price))
        
        # Long: Uptrend + RSI oversold (wave C completion)
        if uptrend and rsi < 35:
            bullish = self.data.Close[-1] > self.data.Open[-1]
            if bullish:
                sl = self.data.Low[-1] * 0.98
                tp = price + (price - sl) * 2.5
                self.buy(size=shares, sl=sl, tp=tp)
        
        # Short: Downtrend + RSI overbought (wave C completion)
        elif downtrend and rsi > 65:
            bearish = self.data.Close[-1] < self.data.Open[-1]
            if bearish:
                sl = self.data.High[-1] * 1.02
                tp = price - (sl - price) * 2.5
                self.sell(size=shares, sl=sl, tp=tp)


if __name__ == '__main__':
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_4h_200weeks.csv')
    
    print("="*60)
    print("WAVE CORRECTION ENTRY - STRATEGY 10")
    print("="*60)
    
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, WaveCorrectionEntry, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    ret = stats['Return [%]'] if pd.notna(stats['Return [%]']) else 0
    sharpe = stats['Sharpe Ratio'] if pd.notna(stats['Sharpe Ratio']) else 0
    
    print(f"Return: {ret:.2f}%, Sharpe: {sharpe:.3f}, Trades: {stats['# Trades']}")
    print("✅ PROFITABLE" if ret > 0 else "❌ NEEDS WORK")
    
    os.makedirs('results', exist_ok=True)
    with open('results/wave_correction_result.json', 'w') as f:
        json.dump({'strategy': 'wave_correction', 'return': float(ret), 'sharpe': float(sharpe), 'trades': int(stats['# Trades'])}, f)
