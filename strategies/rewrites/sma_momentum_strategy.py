"""
SMA Momentum Strategy - Built on Profitable Foundation
========================================================
Based on the profitable SMA 50/200 crossover result on 4H BTC.
Enhancements:
1. Add ATR-based position sizing for risk control
2. Add momentum confirmation (RSI) 
3. Multi-asset testing to validate robustness

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(period).mean().values
    return atr


class SMAMomentumStrategy(Strategy):
    """
    SMA 50/200 crossover with momentum confirmation.
    Based on profitable 4H BTC result.
    
    Entry:
    - LONG: Price > SMA50 > SMA200 + RSI > 50
    - SHORT: Price < SMA50 < SMA200 + RSI < 50
    
    Exit:
    - LONG: Close < SMA50 or RSI < 40
    - SHORT: Close > SMA50 or RSI > 60
    """
    
    sma_fast = 50
    sma_slow = 200
    rsi_period = 14
    atr_period = 14
    risk_per_trade = 0.02
    
    def init(self):
        close = pd.Series(self.data.Close)
        
        sma_fast = ta.sma(close, length=self.sma_fast)
        sma_slow = ta.sma(close, length=self.sma_slow)
        rsi = ta.rsi(close, length=self.rsi_period)
        
        self.sma_fast = self.I(lambda: sma_fast.values)
        self.sma_slow = self.I(lambda: sma_slow.values)
        self.rsi = self.I(lambda: rsi.values)
        
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close,
                         self.atr_period)
    
    def next(self):
        if len(self.data) < 210:
            return
        
        price = self.data.Close[-1]
        sma_f = self.sma_fast[-1]
        sma_s = self.sma_slow[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        
        if np.isnan(sma_f) or np.isnan(sma_s) or np.isnan(rsi):
            return
        
        # Trend
        uptrend = sma_f > sma_s
        downtrend = sma_f < sma_s
        
        # Momentum
        bullish = rsi > 50
        bearish = rsi < 50
        
        # Exit conditions
        if self.position:
            if self.position.is_long:
                if price < sma_f or rsi < 40:
                    self.position.close()
            elif self.position.is_short:
                if price > sma_f or rsi > 60:
                    self.position.close()
            return
        
        # Position sizing
        if np.isnan(atr) or atr <= 0:
            shares = 1
        else:
            risk_amount = self.equity * self.risk_per_trade
            sl_distance = atr * 2
            shares = int(risk_amount / sl_distance) if sl_distance > 0 else 1
        
        max_shares = int(self.equity * 0.5 / price)
        shares = min(shares, max_shares)
        if shares < 1:
            shares = 1
        
        # Entry
        if price > sma_f and uptrend and bullish:
            sl = price - (atr * 2) if not np.isnan(atr) else None
            self.buy(size=shares, sl=sl)
        elif price < sma_f and downtrend and bearish:
            sl = price + (atr * 2) if not np.isnan(atr) else None
            self.sell(size=shares, sl=sl)


def run_multi_asset_test():
    datasets = [
        ("BTC 4H (2021-2025)", "data/crypto/BTC-USDT_4h_200weeks.csv"),
        ("BTC 1H (2021-2025)", "data/crypto/BTC-USDT_1h_200weeks.csv"),
        ("BTC 15m 2025", "data/crypto/BTCUSDT_P_15m_2025.csv"),
        ("ETH 15m", "data/crypto/ETHUSD_15m.csv"),
        ("SPY 15m", "data/equities/SPY_15m.csv"),
        ("QQQ 15m", "data/equities/QQQ_15m.csv"),
    ]
    
    results = []
    
    print("="*70)
    print("SMA MOMENTUM STRATEGY - MULTI-ASSET TEST")
    print("="*70)
    
    for name, path in datasets:
        if not os.path.exists(path):
            print(f"  ❌ {name}: File not found")
            continue
        
        try:
            data = pd.read_csv(path, parse_dates=[0], index_col=0)
            data.columns = [c.strip().capitalize() for c in data.columns]
            
            bt = Backtest(data, SMAMomentumStrategy, cash=1_000_000, 
                         commission=0.002, trade_on_close=True)
            stats = bt.run()
            
            result = {
                'asset': name,
                'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
                'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
                'max_dd': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
                'trades': int(stats['# Trades']),
                'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
            }
            results.append(result)
            
            prefix = "✅" if result['return'] > 0 else "❌"
            print(f"  {prefix} {name}: Return={result['return']:.1f}%, Sharpe={result['sharpe']:.2f}, Trades={result['trades']}, WR={result['win_rate']:.1f}%")
            
        except Exception as e:
            print(f"  ❌ {name}: {str(e)[:50]}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        profitable = [r for r in results if r['return'] > 0]
        print(f"Assets Tested:    {len(results)}")
        print(f"Profitable:       {len(profitable)}/{len(results)}")
        print(f"Avg Return:       {np.mean([r['return'] for r in results]):.2f}%")
        print(f"Avg Sharpe:       {np.mean([r['sharpe'] for r in results if r['sharpe'] != 0]):.3f}")
        
        if profitable:
            print(f"\n🏆 PROFITABLE STRATEGIES:")
            for p in profitable:
                print(f"   - {p['asset']}: {p['return']:.1f}%, Sharpe={p['sharpe']:.2f}")
    
    os.makedirs('results', exist_ok=True)
    with open('results/sma_momentum_multi_asset.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    run_multi_asset_test()
