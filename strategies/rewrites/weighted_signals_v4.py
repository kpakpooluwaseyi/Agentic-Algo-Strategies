"""
Weighted Signals V4 - Back to Simplicity
=========================================
V3 overtraded. V2 was best (43 trades, 53% WR, +0.6%).

V4 approach: Simplify to core mechanics
- Only 3 signals: MACD cross, trend filter, momentum
- Higher bar for entry
- Keep V2's risk management (it worked)

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


class WeightedSignalsV4(Strategy):
    """
    Simplified V4:
    - MACD cross on current bar (strict)
    - EMA trend alignment (EMA20 > EMA50 > EMA200)
    - No RSI/volume complexity
    - Wide stops (2.5x ATR), good R:R (3.5x ATR target)
    """
    
    atr_period = 14
    atr_stop_mult = 2.5
    atr_target_mult = 3.5  # 1.4:1 R:R (lower is often more reliable)
    risk_per_trade = 0.02
    min_move_pct = 0.012
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        self.macd_line = self.I(lambda: macd.iloc[:, 0].values)
        self.macd_signal = self.I(lambda: macd.iloc[:, 1].values)
        self.macd_hist = self.I(lambda: macd.iloc[:, 2].values)
        
        ema_20 = ta.ema(close, length=20)
        ema_50 = ta.ema(close, length=50)
        ema_200 = ta.ema(close, length=200)
        self.ema_20 = self.I(lambda: ema_20.values)
        self.ema_50 = self.I(lambda: ema_50.values)
        self.ema_200 = self.I(lambda: ema_200.values)
        
        self.atr = self.I(calculate_atr, high, low, close, self.atr_period)
    
    def is_macd_bullish_cross(self):
        """Strict MACD bullish cross on current bar."""
        if len(self.macd_line) < 2:
            return False
        return (self.macd_line[-1] > self.macd_signal[-1] and 
                self.macd_line[-2] <= self.macd_signal[-2])
    
    def is_macd_bearish_cross(self):
        """Strict MACD bearish cross on current bar."""
        if len(self.macd_line) < 2:
            return False
        return (self.macd_line[-1] < self.macd_signal[-1] and 
                self.macd_line[-2] >= self.macd_signal[-2])
    
    def is_trend_aligned_up(self):
        """EMA20 > EMA50 > EMA200."""
        if np.isnan(self.ema_20[-1]) or np.isnan(self.ema_50[-1]) or np.isnan(self.ema_200[-1]):
            return False
        return self.ema_20[-1] > self.ema_50[-1] > self.ema_200[-1]
    
    def is_trend_aligned_down(self):
        """EMA20 < EMA50 < EMA200."""
        if np.isnan(self.ema_20[-1]) or np.isnan(self.ema_50[-1]) or np.isnan(self.ema_200[-1]):
            return False
        return self.ema_20[-1] < self.ema_50[-1] < self.ema_200[-1]
    
    def next(self):
        if self.position or len(self.data) < 210:
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        # Simple entry: MACD cross + trend alignment
        go_long = self.is_macd_bullish_cross() and self.is_trend_aligned_up()
        go_short = self.is_macd_bearish_cross() and self.is_trend_aligned_down()
        
        if not (go_long or go_short):
            return
        
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
    print("WEIGHTED SIGNALS V4 - SIMPLIFIED")
    print("Entry: MACD cross + full EMA alignment")
    print("="*70)
    
    datasets = [
        ("data/crypto/BTC-USDT_15m_160weeks.csv", "BTC 15m 2022-2025"),
        ("data/crypto/BTCUSDT_P_15m_2025.csv", "BTC 15m 2025"),
        ("data/crypto/BTC-USDT_1h_200weeks.csv", "BTC 1h 2021-2025"),
    ]
    
    for path, name in datasets:
        if os.path.exists(path):
            data = pd.read_csv(path, parse_dates=[0], index_col=0)
            data.columns = [c.strip().capitalize() for c in data.columns]
            
            bt = Backtest(data, WeightedSignalsV4, cash=1_000_000, 
                         commission=0.002, trade_on_close=True)
            stats = bt.run()
            
            ret = float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0
            sharpe = float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0
            trades = int(stats['# Trades'])
            wr = float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0
            
            s = "✅" if ret > 0 else "❌"
            print(f"\n{name}: {s} Return: {ret:.2f}%, Sharpe: {sharpe:.3f}, Trades: {trades}, WR: {wr:.1f}%")
    
    print("\n" + "="*70)
