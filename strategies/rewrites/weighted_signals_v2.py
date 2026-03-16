"""
Weighted Signals Strategy V2 - Iteration
==========================================
Changes from V1:
1. Higher entry threshold: 55 -> 70 (need more confluence)
2. Wider stops: 1.5x -> 2.5x ATR
3. Added trend filter: EMA50 > EMA200 for longs
4. Adjusted RSI levels: 40 -> 35 for oversold, 60 -> 65 for overbought

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


class WeightedSignalsV2(Strategy):
    """
    V2 Changes:
    - Entry threshold: 70 (was 55)
    - ATR stop: 2.5x (was 1.5x)
    - Added trend filter (EMA50 > EMA200)
    - RSI oversold: 35 (was 40)
    """
    
    # Signal weights
    w_macd = 30
    w_rsi = 25
    w_ema = 20
    w_volume = 15
    w_candle = 10
    
    # CHANGED: Higher threshold
    entry_threshold = 70
    
    # CHANGED: Wider stops
    atr_period = 14
    atr_stop_mult = 2.5
    atr_target_mult = 4.0  # Maintain 1.6:1 R:R
    risk_per_trade = 0.015  # Slightly lower risk
    min_move_pct = 0.015  # 1.5% minimum move
    
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
        
        # EMAs for trend
        ema_20 = ta.ema(close, length=20)
        ema_50 = ta.ema(close, length=50)
        ema_200 = ta.ema(close, length=200)
        self.ema_20 = self.I(lambda: ema_20.values)
        self.ema_50 = self.I(lambda: ema_50.values)
        self.ema_200 = self.I(lambda: ema_200.values)
        
        # Volume
        vol_ma = volume.rolling(20).mean()
        self.vol_ma = self.I(lambda: vol_ma.values)
        
        # ATR
        self.atr = self.I(calculate_atr, high, low, close, self.atr_period)
    
    def get_trend(self):
        """Determine trend using EMA50 vs EMA200."""
        if np.isnan(self.ema_50[-1]) or np.isnan(self.ema_200[-1]):
            return 0  # No trend
        if self.ema_50[-1] > self.ema_200[-1]:
            return 1  # Uptrend
        elif self.ema_50[-1] < self.ema_200[-1]:
            return -1  # Downtrend
        return 0
    
    def calculate_long_score(self):
        score = 0
        
        # MACD bullish cross (30 pts)
        if len(self.macd_line) > 1 and not np.isnan(self.macd_line[-1]):
            macd_cross = self.macd_line[-1] > self.macd_signal[-1] and self.macd_line[-2] <= self.macd_signal[-2]
            if macd_cross:
                score += self.w_macd
        
        # RSI oversold - CHANGED: 35 (was 40)
        if not np.isnan(self.rsi[-1]) and self.rsi[-1] < 35:
            score += self.w_rsi
        
        # Price above EMA20
        if not np.isnan(self.ema_20[-1]) and self.data.Close[-1] > self.ema_20[-1]:
            score += self.w_ema
        
        # Volume spike
        if not np.isnan(self.vol_ma[-1]) and self.data.Volume[-1] > self.vol_ma[-1] * 1.2:
            score += self.w_volume
        
        # Bullish candle (body > 50% of range)
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        range_ = self.data.High[-1] - self.data.Low[-1]
        if range_ > 0 and body / range_ > 0.5 and self.data.Close[-1] > self.data.Open[-1]:
            score += self.w_candle
        
        return score
    
    def calculate_short_score(self):
        score = 0
        
        # MACD bearish cross
        if len(self.macd_line) > 1 and not np.isnan(self.macd_line[-1]):
            macd_cross = self.macd_line[-1] < self.macd_signal[-1] and self.macd_line[-2] >= self.macd_signal[-2]
            if macd_cross:
                score += self.w_macd
        
        # RSI overbought - CHANGED: 65 (was 60)
        if not np.isnan(self.rsi[-1]) and self.rsi[-1] > 65:
            score += self.w_rsi
        
        # Price below EMA20
        if not np.isnan(self.ema_20[-1]) and self.data.Close[-1] < self.ema_20[-1]:
            score += self.w_ema
        
        # Volume spike
        if not np.isnan(self.vol_ma[-1]) and self.data.Volume[-1] > self.vol_ma[-1] * 1.2:
            score += self.w_volume
        
        # Bearish candle
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
        
        # NEW: Trend filter
        trend = self.get_trend()
        
        # Calculate scores
        long_score = self.calculate_long_score()
        short_score = self.calculate_short_score()
        
        # Only trade with trend
        go_long = long_score >= self.entry_threshold and trend >= 0  # Up or neutral
        go_short = short_score >= self.entry_threshold and trend <= 0  # Down or neutral
        
        if not (go_long or go_short):
            return
        
        # If both valid, take higher score
        if go_long and go_short:
            if long_score > short_score:
                go_short = False
            else:
                go_long = False
        
        # Calculate stops and targets
        stop_distance = atr * self.atr_stop_mult
        target_distance = atr * self.atr_target_mult
        
        # Commission protection
        expected_move_pct = target_distance / price
        if expected_move_pct < self.min_move_pct:
            return
        
        # Position sizing
        risk_amount = self.equity * self.risk_per_trade
        shares = int(risk_amount / stop_distance)
        max_shares = int(self.equity * 0.3 / price)
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
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, WeightedSignalsV2, cash=1_000_000, 
                 commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    return {
        'name': name,
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'trades': int(stats['# Trades']),
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        'max_dd': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0
    }


if __name__ == '__main__':
    print("="*70)
    print("WEIGHTED SIGNALS V2 - ITERATION TEST")
    print("Changes: threshold 70, ATR 2.5x, trend filter, stricter RSI")
    print("="*70)
    
    datasets = [
        ("data/crypto/BTC-USDT_15m_160weeks.csv", "BTC 15m (2022-2025)"),
        ("data/crypto/BTCUSDT_P_15m_2025.csv", "BTC 15m (2025 only)"),
    ]
    
    results = []
    
    for path, name in datasets:
        if os.path.exists(path):
            print(f"\nTesting {name}...")
            r = run_test(path, name)
            results.append(r)
            
            status = "✅" if r['return'] > 0 else "❌"
            print(f"  {status} Return: {r['return']:.2f}%, Sharpe: {r['sharpe']:.3f}, "
                  f"Trades: {r['trades']}, WR: {r['win_rate']:.1f}%")
    
    print("\n" + "="*70)
    print("V2 SUMMARY")
    print("="*70)
    print(f"Profitable: {len([r for r in results if r['return'] > 0])}/{len(results)}")
    
    os.makedirs('results', exist_ok=True)
    with open('results/weighted_signals_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
