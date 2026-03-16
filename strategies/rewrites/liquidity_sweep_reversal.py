"""
Strategy 5: Liquidity Sweep Reversal
=====================================
Detect false breakouts (liquidity sweeps) and trade reversals.

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


class LiquiditySweepReversal(Strategy):
    """
    Detects liquidity sweeps (false breakouts) and trades reversals.
    Long: Price pierces below support then closes back above
    Short: Price pierces above resistance then closes back below
    """
    
    lookback = 20
    risk_per_trade = 0.02
    
    def init(self):
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Support/Resistance levels
        self.resistance = self.I(lambda: high.shift(1).rolling(self.lookback).max().values)
        self.support = self.I(lambda: low.shift(1).rolling(self.lookback).min().values)
        
        self.atr = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, 14)
    
    def next(self):
        if self.position or len(self.data) < 30:
            return
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        resistance = self.resistance[-1]
        support = self.support[-1]
        
        if np.isnan(resistance) or np.isnan(support):
            return
        
        # Liquidity sweep long: wick below support, close above
        swept_support = low < support and price > support
        
        # Liquidity sweep short: wick above resistance, close below
        swept_resistance = high > resistance and price < resistance
        
        shares = max(1, min(int((self.equity * self.risk_per_trade) / (atr * 2)),
                           int(self.equity * 0.5 / price)))
        
        if swept_support:
            sl = low - (atr * 0.5)
            tp = price + (atr * 3)
            self.buy(size=shares, sl=sl, tp=tp)
        
        elif swept_resistance:
            sl = high + (atr * 0.5)
            tp = price - (atr * 3)
            self.sell(size=shares, sl=sl, tp=tp)


if __name__ == '__main__':
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_4h_200weeks.csv')
    
    print("="*60)
    print("LIQUIDITY SWEEP REVERSAL - STRATEGY 5")
    print("="*60)
    
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, LiquiditySweepReversal, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    ret = stats['Return [%]'] if pd.notna(stats['Return [%]']) else 0
    sharpe = stats['Sharpe Ratio'] if pd.notna(stats['Sharpe Ratio']) else 0
    
    print(f"Return: {ret:.2f}%, Sharpe: {sharpe:.3f}, Trades: {stats['# Trades']}")
    print("✅ PROFITABLE" if ret > 0 else "❌ NEEDS WORK")
    
    os.makedirs('results', exist_ok=True)
    with open('results/liquidity_sweep_result.json', 'w') as f:
        json.dump({'strategy': 'liquidity_sweep', 'return': float(ret), 'sharpe': float(sharpe), 'trades': int(stats['# Trades'])}, f)
