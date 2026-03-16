"""
Walk-Forward Analysis for Weighted Signals V2
===============================================
Tests if V2 is overfit by using rolling in-sample/out-of-sample windows.

Method:
- Split data into 6-month windows
- Train on first 4 months, test on last 2 months
- Roll forward and repeat
- Compare IS vs OOS performance

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json
from datetime import timedelta


def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    return pd.Series(tr).rolling(period).mean().values


class WeightedSignalsV2(Strategy):
    """Copy of V2 for WFA testing."""
    
    w_macd = 30
    w_rsi = 25
    w_ema = 20
    w_volume = 15
    w_candle = 10
    entry_threshold = 70
    atr_period = 14
    atr_stop_mult = 2.5
    atr_target_mult = 4.0
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
        
        self.atr = self.I(calculate_atr, high, low, close, self.atr_period)
    
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


def run_wfa(data_path, window_months=6, oos_months=2):
    """Run Walk-Forward Analysis."""
    print(f"\nLoading {data_path}...")
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    start = data.index.min()
    end = data.index.max()
    
    results = []
    current = start
    window_num = 0
    
    print(f"Data range: {start.date()} to {end.date()}")
    print(f"Window: {window_months}mo IS, {oos_months}mo OOS")
    print("-" * 60)
    
    while current + pd.DateOffset(months=window_months) < end:
        window_num += 1
        is_start = current
        is_end = current + pd.DateOffset(months=window_months - oos_months)
        oos_start = is_end
        oos_end = current + pd.DateOffset(months=window_months)
        
        # Get data slices
        is_data = data[is_start:is_end]
        oos_data = data[oos_start:oos_end]
        
        if len(is_data) < 1000 or len(oos_data) < 500:
            current += pd.DateOffset(months=oos_months)
            continue
        
        try:
            # In-sample backtest
            bt_is = Backtest(is_data, WeightedSignalsV2, cash=1_000_000, commission=0.002, trade_on_close=True)
            stats_is = bt_is.run()
            ret_is = float(stats_is['Return [%]']) if pd.notna(stats_is['Return [%]']) else 0
            
            # Out-of-sample backtest
            bt_oos = Backtest(oos_data, WeightedSignalsV2, cash=1_000_000, commission=0.002, trade_on_close=True)
            stats_oos = bt_oos.run()
            ret_oos = float(stats_oos['Return [%]']) if pd.notna(stats_oos['Return [%]']) else 0
            trades_oos = int(stats_oos['# Trades'])
            
            results.append({
                'window': window_num,
                'is_start': str(is_start.date()),
                'is_end': str(is_end.date()),
                'oos_start': str(oos_start.date()),
                'oos_end': str(oos_end.date()),
                'is_return': ret_is,
                'oos_return': ret_oos,
                'oos_trades': trades_oos,
                'degradation': ret_is - ret_oos
            })
            
            status = "✅" if ret_oos > 0 else "❌"
            print(f"W{window_num}: IS={ret_is:+.2f}% | OOS={ret_oos:+.2f}% {status} | Trades={trades_oos}")
            
        except Exception as e:
            print(f"W{window_num}: Error - {str(e)[:50]}")
        
        current += pd.DateOffset(months=oos_months)
    
    return results


if __name__ == '__main__':
    print("="*70)
    print("WALK-FORWARD ANALYSIS - WEIGHTED SIGNALS V2")
    print("="*70)
    
    data_path = "data/crypto/BTC-USDT_15m_160weeks.csv"
    
    results = run_wfa(data_path, window_months=6, oos_months=2)
    
    if results:
        print("\n" + "="*70)
        print("WFA SUMMARY")
        print("="*70)
        
        oos_returns = [r['oos_return'] for r in results]
        profitable_windows = sum(1 for r in oos_returns if r > 0)
        avg_oos = np.mean(oos_returns)
        avg_degradation = np.mean([r['degradation'] for r in results])
        
        print(f"Total windows: {len(results)}")
        print(f"Profitable OOS: {profitable_windows}/{len(results)} ({100*profitable_windows/len(results):.0f}%)")
        print(f"Avg OOS return: {avg_oos:.2f}%")
        print(f"Avg degradation (IS-OOS): {avg_degradation:.2f}%")
        
        # Verdict
        if profitable_windows / len(results) >= 0.5 and avg_oos > -5:
            print("\n✅ VERDICT: Strategy shows robustness (not heavily overfit)")
        else:
            print("\n❌ VERDICT: Strategy may be overfit (poor OOS performance)")
        
        # Save
        os.makedirs('results', exist_ok=True)
        with open('results/v2_wfa_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to results/v2_wfa_results.json")
