"""
Weighted Signals Strategy - Lower Timeframe Prototype
======================================================
Implements all 3 solutions from the framework:
1. Weighted signal scoring (not binary AND)
2. ATR-based dynamic stops
3. Commission protection filter

Target: 15m BTC data

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
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
    return pd.Series(tr).rolling(period).mean().values


class WeightedSignalsStrategy(Strategy):
    """
    Weighted Signals Strategy
    
    Instead of requiring all conditions (AND logic), assigns weights
    to each signal and enters when total score exceeds threshold.
    
    Signal Weights (Long):
    - MACD bullish cross: 30 points
    - RSI < 40 (oversold): 25 points  
    - Price above EMA20: 20 points
    - Volume above average: 15 points
    - Bullish candle: 10 points
    
    Entry: Score >= 55 (need ~2-3 signals, not all 5)
    
    Risk Management:
    - Stop: 1.5x ATR
    - Target: 3x ATR (2:1 R:R)
    - Skip if expected move < 1.2% (commission protection)
    """
    
    # Signal weights
    w_macd = 30
    w_rsi = 25
    w_ema = 20
    w_volume = 15
    w_candle = 10
    
    # Entry threshold (out of 100)
    entry_threshold = 55
    
    # Risk parameters
    atr_period = 14
    atr_stop_mult = 1.5
    atr_target_mult = 3.0
    risk_per_trade = 0.02
    min_move_pct = 0.012  # 1.2% minimum move (3x commission)
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # MACD
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        self.macd_line = self.I(lambda: macd.iloc[:, 0].values)
        self.macd_signal = self.I(lambda: macd.iloc[:, 1].values)
        
        # RSI
        rsi = ta.rsi(close, length=14)
        self.rsi = self.I(lambda: rsi.values)
        
        # EMA
        ema_20 = ta.ema(close, length=20)
        ema_50 = ta.ema(close, length=50)
        self.ema_20 = self.I(lambda: ema_20.values)
        self.ema_50 = self.I(lambda: ema_50.values)
        
        # Volume MA
        vol_ma = volume.rolling(20).mean()
        self.vol_ma = self.I(lambda: vol_ma.values)
        
        # ATR
        self.atr = self.I(calculate_atr, high, low, close, self.atr_period)
    
    def calculate_long_score(self):
        """Calculate weighted score for long entry."""
        score = 0
        
        # MACD bullish cross (30 pts)
        if len(self.macd_line) > 1:
            macd_cross = self.macd_line[-1] > self.macd_signal[-1] and self.macd_line[-2] <= self.macd_signal[-2]
            if macd_cross:
                score += self.w_macd
        
        # RSI oversold (25 pts)
        if not np.isnan(self.rsi[-1]) and self.rsi[-1] < 40:
            score += self.w_rsi
        
        # Price above EMA20 (20 pts)
        if not np.isnan(self.ema_20[-1]) and self.data.Close[-1] > self.ema_20[-1]:
            score += self.w_ema
        
        # Volume above average (15 pts)
        if not np.isnan(self.vol_ma[-1]) and self.data.Volume[-1] > self.vol_ma[-1]:
            score += self.w_volume
        
        # Bullish candle (10 pts)
        if self.data.Close[-1] > self.data.Open[-1]:
            score += self.w_candle
        
        return score
    
    def calculate_short_score(self):
        """Calculate weighted score for short entry."""
        score = 0
        
        # MACD bearish cross (30 pts)
        if len(self.macd_line) > 1:
            macd_cross = self.macd_line[-1] < self.macd_signal[-1] and self.macd_line[-2] >= self.macd_signal[-2]
            if macd_cross:
                score += self.w_macd
        
        # RSI overbought (25 pts)
        if not np.isnan(self.rsi[-1]) and self.rsi[-1] > 60:
            score += self.w_rsi
        
        # Price below EMA20 (20 pts)
        if not np.isnan(self.ema_20[-1]) and self.data.Close[-1] < self.ema_20[-1]:
            score += self.w_ema
        
        # Volume above average (15 pts)
        if not np.isnan(self.vol_ma[-1]) and self.data.Volume[-1] > self.vol_ma[-1]:
            score += self.w_volume
        
        # Bearish candle (10 pts)
        if self.data.Close[-1] < self.data.Open[-1]:
            score += self.w_candle
        
        return score
    
    def next(self):
        if self.position:
            return
        if len(self.data) < 30:
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        # Calculate scores
        long_score = self.calculate_long_score()
        short_score = self.calculate_short_score()
        
        # Determine direction (take higher score if both above threshold)
        go_long = long_score >= self.entry_threshold and long_score > short_score
        go_short = short_score >= self.entry_threshold and short_score > long_score
        
        if not (go_long or go_short):
            return
        
        # Calculate stops and targets
        stop_distance = atr * self.atr_stop_mult
        target_distance = atr * self.atr_target_mult
        
        # COMMISSION PROTECTION: Skip if move too small
        expected_move_pct = target_distance / price
        if expected_move_pct < self.min_move_pct:
            return  # Not worth the commission
        
        # Position sizing
        risk_amount = self.equity * self.risk_per_trade
        shares = int(risk_amount / stop_distance)
        max_shares = int(self.equity * 0.5 / price)
        shares = max(1, min(shares, max_shares))
        
        if go_long:
            sl = price - stop_distance
            tp = price + target_distance
            self.buy(size=shares, sl=sl, tp=tp)
        
        elif go_short:
            sl = price + stop_distance
            tp = price - target_distance
            self.sell(size=shares, sl=sl, tp=tp)


def run_test(data_path, name):
    """Run backtest and return results."""
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, WeightedSignalsStrategy, cash=1_000_000, 
                 commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    ret = float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0
    sharpe = float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0
    trades = int(stats['# Trades'])
    win_rate = float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0
    max_dd = float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0
    
    return {
        'name': name,
        'return': ret,
        'sharpe': sharpe,
        'trades': trades,
        'win_rate': win_rate,
        'max_dd': max_dd
    }


if __name__ == '__main__':
    print("="*70)
    print("WEIGHTED SIGNALS STRATEGY - LOWER TIMEFRAME TEST")
    print("="*70)
    
    # Test on multiple 15m datasets
    datasets = [
        ("data/crypto/BTC-USDT_15m_160weeks.csv", "BTC 15m (2022-2025)"),
        ("data/crypto/BTCUSDT_P_15m_2025.csv", "BTC 15m (2025 only)"),
        ("data/crypto/ETHUSD_15m.csv", "ETH 15m"),
    ]
    
    results = []
    
    for path, name in datasets:
        if os.path.exists(path):
            print(f"\nTesting on {name}...")
            r = run_test(path, name)
            results.append(r)
            
            status = "✅" if r['return'] > 0 else "❌"
            print(f"  {status} Return: {r['return']:.2f}%, Sharpe: {r['sharpe']:.3f}, "
                  f"Trades: {r['trades']}, WR: {r['win_rate']:.1f}%, MaxDD: {r['max_dd']:.1f}%")
        else:
            print(f"  ⚠️ {name}: File not found")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    profitable = [r for r in results if r['return'] > 0]
    print(f"Profitable: {len(profitable)}/{len(results)}")
    print(f"Avg Return: {np.mean([r['return'] for r in results]):.2f}%")
    print(f"Avg Sharpe: {np.mean([r['sharpe'] for r in results]):.3f}")
    print(f"Avg Trades: {np.mean([r['trades'] for r in results]):.0f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/weighted_signals_v1.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to results/weighted_signals_v1.json")
