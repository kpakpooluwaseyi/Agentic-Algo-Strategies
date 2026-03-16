"""
Parameter Optimization for Weighted Signals V2
================================================
Optimize key parameters to improve from +0.6% baseline.

Parameters to optimize:
- entry_threshold: [60, 65, 70, 75, 80]
- atr_stop_mult: [2.0, 2.5, 3.0]
- atr_target_mult: [3.0, 4.0, 5.0, 6.0]

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json
from itertools import product
import warnings
warnings.filterwarnings('ignore')


def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    return pd.Series(tr).rolling(period).mean().values


class WeightedSignalsOpt(Strategy):
    """Optimizable version of V2."""
    
    w_macd = 30
    w_rsi = 25
    w_ema = 20
    w_volume = 15
    w_candle = 10
    entry_threshold = 70  # Will be optimized
    atr_stop_mult = 2.5   # Will be optimized
    atr_target_mult = 4.0 # Will be optimized
    risk_per_trade = 0.015
    min_move_pct = 0.015
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        self.macd_line = self.I(lambda: macd.iloc[:, 0].values)
        self.macd_signal = self.I(lambda: macd.iloc[:, 1].values)
        
        rsi = ta.rsi(close, length=14)
        self.rsi = self.I(lambda: rsi.values)
        
        ema_20 = ta.ema(close, length=20)
        ema_50 = ta.ema(close, length=50)
        ema_200 = ta.ema(close, length=200)
        self.ema_20 = self.I(lambda: ema_20.values)
        self.ema_50 = self.I(lambda: ema_50.values)
        self.ema_200 = self.I(lambda: ema_200.values)
        
        vol_ma = volume.rolling(20).mean()
        self.vol_ma = self.I(lambda: vol_ma.values)
        
        self.atr = self.I(calculate_atr, high, low, close, 14)
    
    def get_trend(self):
        if np.isnan(self.ema_50[-1]) or np.isnan(self.ema_200[-1]):
            return 0
        if self.ema_50[-1] > self.ema_200[-1]:
            return 1
        elif self.ema_50[-1] < self.ema_200[-1]:
            return -1
        return 0
    
    def calculate_long_score(self):
        score = 0
        if len(self.macd_line) > 1 and not np.isnan(self.macd_line[-1]):
            if self.macd_line[-1] > self.macd_signal[-1] and self.macd_line[-2] <= self.macd_signal[-2]:
                score += self.w_macd
        if not np.isnan(self.rsi[-1]) and self.rsi[-1] < 35:
            score += self.w_rsi
        if not np.isnan(self.ema_20[-1]) and self.data.Close[-1] > self.ema_20[-1]:
            score += self.w_ema
        if not np.isnan(self.vol_ma[-1]) and self.data.Volume[-1] > self.vol_ma[-1] * 1.2:
            score += self.w_volume
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        range_ = self.data.High[-1] - self.data.Low[-1]
        if range_ > 0 and body / range_ > 0.5 and self.data.Close[-1] > self.data.Open[-1]:
            score += self.w_candle
        return score
    
    def calculate_short_score(self):
        score = 0
        if len(self.macd_line) > 1 and not np.isnan(self.macd_line[-1]):
            if self.macd_line[-1] < self.macd_signal[-1] and self.macd_line[-2] >= self.macd_signal[-2]:
                score += self.w_macd
        if not np.isnan(self.rsi[-1]) and self.rsi[-1] > 65:
            score += self.w_rsi
        if not np.isnan(self.ema_20[-1]) and self.data.Close[-1] < self.ema_20[-1]:
            score += self.w_ema
        if not np.isnan(self.vol_ma[-1]) and self.data.Volume[-1] > self.vol_ma[-1] * 1.2:
            score += self.w_volume
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        range_ = self.data.High[-1] - self.data.Low[-1]
        if range_ > 0 and body / range_ > 0.5 and self.data.Close[-1] < self.data.Open[-1]:
            score += self.w_candle
        return score
    
    def next(self):
        if self.position or len(self.data) < 210:
            return
        price = self.data.Close[-1]
        atr = self.atr[-1]
        if np.isnan(atr) or atr <= 0:
            return
        trend = self.get_trend()
        long_score = self.calculate_long_score()
        short_score = self.calculate_short_score()
        go_long = long_score >= self.entry_threshold and trend >= 0
        go_short = short_score >= self.entry_threshold and trend <= 0
        if not (go_long or go_short):
            return
        if go_long and go_short:
            go_short = False if long_score > short_score else True
            go_long = not go_short
        stop_distance = atr * self.atr_stop_mult
        target_distance = atr * self.atr_target_mult
        if target_distance / price < self.min_move_pct:
            return
        risk_amount = self.equity * self.risk_per_trade
        shares = max(1, min(int(risk_amount / stop_distance), int(self.equity * 0.3 / price)))
        if go_long:
            self.buy(size=shares, sl=price - stop_distance, tp=price + target_distance)
        elif go_short:
            self.sell(size=shares, sl=price + stop_distance, tp=price - target_distance)


if __name__ == '__main__':
    print("="*70)
    print("PARAMETER OPTIMIZATION - WEIGHTED SIGNALS V2")
    print("="*70)
    
    data_path = "data/crypto/BTC-USDT_15m_160weeks.csv"
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    # Parameter grid
    thresholds = [60, 65, 70, 75, 80]
    stop_mults = [2.0, 2.5, 3.0]
    target_mults = [3.0, 4.0, 5.0, 6.0]
    
    total = len(thresholds) * len(stop_mults) * len(target_mults)
    print(f"Testing {total} combinations...")
    print("-" * 60)
    
    results = []
    best_sharpe = -999
    best_params = None
    
    for i, (threshold, stop_m, target_m) in enumerate(product(thresholds, stop_mults, target_mults)):
        try:
            bt = Backtest(data, WeightedSignalsOpt, cash=1_000_000, commission=0.002, trade_on_close=True)
            stats = bt.run(entry_threshold=threshold, atr_stop_mult=stop_m, atr_target_mult=target_m)
            
            ret = float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0
            sharpe = float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else -999
            trades = int(stats['# Trades'])
            wr = float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0
            
            result = {
                'threshold': threshold,
                'stop_mult': stop_m,
                'target_mult': target_m,
                'return': ret,
                'sharpe': sharpe,
                'trades': trades,
                'win_rate': wr
            }
            results.append(result)
            
            if sharpe > best_sharpe and trades >= 10:
                best_sharpe = sharpe
                best_params = result
                print(f"[{i+1}/{total}] NEW BEST: thresh={threshold}, stop={stop_m}, target={target_m} -> Sharpe={sharpe:.3f}, Ret={ret:.2f}%")
                
        except Exception as e:
            pass
    
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    if best_params:
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Threshold: {best_params['threshold']}")
        print(f"   Stop ATR: {best_params['stop_mult']}x")
        print(f"   Target ATR: {best_params['target_mult']}x")
        print(f"   Return: {best_params['return']:.2f}%")
        print(f"   Sharpe: {best_params['sharpe']:.3f}")
        print(f"   Trades: {best_params['trades']}")
        print(f"   Win Rate: {best_params['win_rate']:.1f}%")
    
    # Top 5
    sorted_results = sorted(results, key=lambda x: x['sharpe'] if x['trades'] >= 10 else -999, reverse=True)[:5]
    
    print("\nTop 5 Configurations:")
    for i, r in enumerate(sorted_results):
        print(f"  {i+1}. thresh={r['threshold']}, stop={r['stop_mult']}, target={r['target_mult']} -> "
              f"Sharpe={r['sharpe']:.3f}, Ret={r['return']:.2f}%, Trades={r['trades']}")
    
    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/v2_optimization_results.json', 'w') as f:
        json.dump({'best': best_params, 'all': results}, f, indent=2)
    print("\nResults saved to results/v2_optimization_results.json")
