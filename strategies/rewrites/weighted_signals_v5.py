"""
Weighted Signals V5 - Optimizing V2
====================================
V2 was profitable (+0.6%, 43 trades, 53% WR).
V3/V4 regressed by overtrading.

V5 changes:
- Keep V2's core (threshold 70, ATR 2.5x stop)
- Add: Only trade when histogram momentum matches
- Add: Require RSI not in neutral zone (clearer signal)
- Try: Higher target (5x ATR) for better R:R

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


class WeightedSignalsV5(Strategy):
    """
    V5 = V2 + histogram momentum + better R:R
    """
    
    w_macd = 30
    w_rsi = 25
    w_ema = 20
    w_volume = 15
    w_candle = 10
    
    # Keep V2's winning threshold
    entry_threshold = 70
    
    # V2's stops + higher target
    atr_period = 14
    atr_stop_mult = 2.5
    atr_target_mult = 5.0  # Was 4.0 in V2, try 5.0 for 2:1 R:R
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
        self.macd_hist = self.I(lambda: macd.iloc[:, 2].values)
        
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
    
    def hist_momentum_up(self):
        """MACD histogram increasing."""
        if len(self.macd_hist) < 2:
            return False
        return self.macd_hist[-1] > self.macd_hist[-2]
    
    def hist_momentum_down(self):
        """MACD histogram decreasing."""
        if len(self.macd_hist) < 2:
            return False
        return self.macd_hist[-1] < self.macd_hist[-2]
    
    def calculate_long_score(self):
        score = 0
        
        # MACD bullish cross
        if len(self.macd_line) > 1 and not np.isnan(self.macd_line[-1]):
            if self.macd_line[-1] > self.macd_signal[-1] and self.macd_line[-2] <= self.macd_signal[-2]:
                score += self.w_macd
        
        # RSI oversold (strict)
        if not np.isnan(self.rsi[-1]) and self.rsi[-1] < 35:
            score += self.w_rsi
        
        # Price above EMA20
        if not np.isnan(self.ema_20[-1]) and self.data.Close[-1] > self.ema_20[-1]:
            score += self.w_ema
        
        # Volume spike (1.2x average)
        if not np.isnan(self.vol_ma[-1]) and self.data.Volume[-1] > self.vol_ma[-1] * 1.2:
            score += self.w_volume
        
        # Bullish candle
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        range_ = self.data.High[-1] - self.data.Low[-1]
        if range_ > 0 and body / range_ > 0.5 and self.data.Close[-1] > self.data.Open[-1]:
            score += self.w_candle
        
        return score
    
    def calculate_short_score(self):
        score = 0
        
        # MACD bearish cross
        if len(self.macd_line) > 1 and not np.isnan(self.macd_line[-1]):
            if self.macd_line[-1] < self.macd_signal[-1] and self.macd_line[-2] >= self.macd_signal[-2]:
                score += self.w_macd
        
        # RSI overbought (strict)
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
        
        trend = self.get_trend()
        long_score = self.calculate_long_score()
        short_score = self.calculate_short_score()
        
        # Entry with trend + histogram momentum confirmation
        go_long = (long_score >= self.entry_threshold and 
                   trend >= 0 and 
                   self.hist_momentum_up())
        go_short = (short_score >= self.entry_threshold and 
                    trend <= 0 and 
                    self.hist_momentum_down())
        
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
    print("WEIGHTED SIGNALS V5 = V2 + Histogram Momentum + 2:1 R:R")
    print("="*70)
    
    datasets = [
        ("data/crypto/BTC-USDT_15m_160weeks.csv", "BTC 15m 2022-2025"),
        ("data/crypto/BTCUSDT_P_15m_2025.csv", "BTC 15m 2025"),
    ]
    
    for path, name in datasets:
        if os.path.exists(path):
            data = pd.read_csv(path, parse_dates=[0], index_col=0)
            data.columns = [c.strip().capitalize() for c in data.columns]
            
            bt = Backtest(data, WeightedSignalsV5, cash=1_000_000, commission=0.002, trade_on_close=True)
            stats = bt.run()
            
            ret = float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0
            sharpe = float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0
            trades = int(stats['# Trades'])
            wr = float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0
            
            s = "✅" if ret > 0 else "❌"
            print(f"\n{name}: {s} Return: {ret:.2f}%, Sharpe: {sharpe:.3f}, Trades: {trades}, WR: {wr:.1f}%")
    
    print("\n" + "="*70)
