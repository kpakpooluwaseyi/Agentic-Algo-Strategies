"""
Strategy 6: Bollinger Band Mean Reversion
==========================================
Classic mean reversion using Bollinger Bands.

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json


class BBMeanReversion(Strategy):
    """
    Long: Price touches lower band + RSI oversold
    Short: Price touches upper band + RSI overbought
    Exit at middle band
    """
    
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    risk_per_trade = 0.02
    
    def init(self):
        close = pd.Series(self.data.Close)
        
        # Bollinger Bands
        bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        self.bb_upper = self.I(lambda: bb.iloc[:, 0].values)  # Upper
        self.bb_middle = self.I(lambda: bb.iloc[:, 1].values)  # Middle
        self.bb_lower = self.I(lambda: bb.iloc[:, 2].values)  # Lower
        
        # RSI
        rsi = ta.rsi(close, length=self.rsi_period)
        self.rsi = self.I(lambda: rsi.values)
    
    def next(self):
        if len(self.data) < 30:
            return
        
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        
        # Exit at middle band
        if self.position:
            if self.position.is_long and price >= self.bb_middle[-1]:
                self.position.close()
            elif self.position.is_short and price <= self.bb_middle[-1]:
                self.position.close()
            return
        
        if np.isnan(rsi) or np.isnan(self.bb_lower[-1]):
            return
        
        shares = max(1, int(self.equity * self.risk_per_trade / price))
        
        # Long: touch lower band + RSI oversold
        if price <= self.bb_lower[-1] and rsi < self.rsi_oversold:
            sl = price * 0.97  # 3% stop
            self.buy(size=shares, sl=sl)
        
        # Short: touch upper band + RSI overbought
        elif price >= self.bb_upper[-1] and rsi > self.rsi_overbought:
            sl = price * 1.03  # 3% stop
            self.sell(size=shares, sl=sl)


if __name__ == '__main__':
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_4h_200weeks.csv')
    
    print("="*60)
    print("BB MEAN REVERSION - STRATEGY 6")
    print("="*60)
    
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, BBMeanReversion, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    ret = stats['Return [%]'] if pd.notna(stats['Return [%]']) else 0
    sharpe = stats['Sharpe Ratio'] if pd.notna(stats['Sharpe Ratio']) else 0
    
    print(f"Return: {ret:.2f}%, Sharpe: {sharpe:.3f}, Trades: {stats['# Trades']}")
    print("✅ PROFITABLE" if ret > 0 else "❌ NEEDS WORK")
    
    os.makedirs('results', exist_ok=True)
    with open('results/bb_mean_reversion_result.json', 'w') as f:
        json.dump({'strategy': 'bb_mean_reversion', 'return': float(ret), 'sharpe': float(sharpe), 'trades': int(stats['# Trades'])}, f)
