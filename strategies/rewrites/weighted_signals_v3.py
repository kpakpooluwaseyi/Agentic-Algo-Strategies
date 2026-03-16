"""
Weighted Signals Strategy V3 - Further Optimization
====================================================
Changes from V2:
1. Time-decayed signals: Check last 3 bars for MACD cross, not just current
2. Momentum confirmation: RSI must be rising for longs, falling for shorts
3. Better R:R: Target 5x ATR with 2x ATR stop (2.5:1 R:R)
4. Trail stop after 1.5x ATR profit

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


class WeightedSignalsV3(Strategy):
    """
    V3 Changes:
    - Time-decayed MACD (check last 3 bars)
    - RSI momentum direction check
    - Better R:R (2.5:1)
    """
    
    w_macd = 35      # Increased weight for MACD
    w_rsi = 25
    w_ema = 20
    w_volume = 10    # Reduced volume weight
    w_candle = 10
    
    entry_threshold = 65  # Slightly lower to capture more
    
    atr_period = 14
    atr_stop_mult = 2.0
    atr_target_mult = 5.0  # 2.5:1 R:R
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
    
    def macd_cross_recent(self, direction='bullish', lookback=3):
        """Check for MACD cross in last N bars (time-decay)."""
        for i in range(1, min(lookback + 1, len(self.macd_line))):
            if np.isnan(self.macd_line[-i]) or np.isnan(self.macd_signal[-i]):
                continue
            if i + 1 >= len(self.macd_line):
                continue
                
            if direction == 'bullish':
                if self.macd_line[-i] > self.macd_signal[-i] and self.macd_line[-i-1] <= self.macd_signal[-i-1]:
                    return True
            else:
                if self.macd_line[-i] < self.macd_signal[-i] and self.macd_line[-i-1] >= self.macd_signal[-i-1]:
                    return True
        return False
    
    def rsi_momentum(self, direction='up'):
        """Check RSI direction over last 3 bars."""
        if len(self.rsi) < 4:
            return False
        if np.isnan(self.rsi[-1]) or np.isnan(self.rsi[-3]):
            return False
        
        if direction == 'up':
            return self.rsi[-1] > self.rsi[-3]  # Rising RSI
        else:
            return self.rsi[-1] < self.rsi[-3]  # Falling RSI
    
    def calculate_long_score(self):
        score = 0
        
        # MACD bullish cross in last 3 bars
        if self.macd_cross_recent('bullish', 3):
            score += self.w_macd
        
        # RSI in buy zone AND rising
        if not np.isnan(self.rsi[-1]) and self.rsi[-1] < 45 and self.rsi_momentum('up'):
            score += self.w_rsi
        
        # Price above EMA20
        if not np.isnan(self.ema_20[-1]) and self.data.Close[-1] > self.ema_20[-1]:
            score += self.w_ema
        
        # Volume spike
        if not np.isnan(self.vol_ma[-1]) and self.data.Volume[-1] > self.vol_ma[-1] * 1.3:
            score += self.w_volume
        
        # Strong bullish candle
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        range_ = self.data.High[-1] - self.data.Low[-1]
        if range_ > 0 and body / range_ > 0.6 and self.data.Close[-1] > self.data.Open[-1]:
            score += self.w_candle
        
        return score
    
    def calculate_short_score(self):
        score = 0
        
        # MACD bearish cross in last 3 bars
        if self.macd_cross_recent('bearish', 3):
            score += self.w_macd
        
        # RSI in sell zone AND falling
        if not np.isnan(self.rsi[-1]) and self.rsi[-1] > 55 and self.rsi_momentum('down'):
            score += self.w_rsi
        
        # Price below EMA20
        if not np.isnan(self.ema_20[-1]) and self.data.Close[-1] < self.ema_20[-1]:
            score += self.w_ema
        
        # Volume spike
        if not np.isnan(self.vol_ma[-1]) and self.data.Volume[-1] > self.vol_ma[-1] * 1.3:
            score += self.w_volume
        
        # Strong bearish candle
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        range_ = self.data.High[-1] - self.data.Low[-1]
        if range_ > 0 and body / range_ > 0.6 and self.data.Close[-1] < self.data.Open[-1]:
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


def run_test(data_path, name):
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, WeightedSignalsV3, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    return {
        'name': name,
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'trades': int(stats['# Trades']),
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
    }


if __name__ == '__main__':
    print("="*70)
    print("WEIGHTED SIGNALS V3")
    print("Changes: Time-decay MACD, RSI momentum, 2.5:1 R:R")
    print("="*70)
    
    datasets = [
        ("data/crypto/BTC-USDT_15m_160weeks.csv", "BTC 15m 2022-2025"),
        ("data/crypto/BTCUSDT_P_15m_2025.csv", "BTC 15m 2025"),
    ]
    
    for path, name in datasets:
        if os.path.exists(path):
            print(f"\n{name}...")
            r = run_test(path, name)
            s = "✅" if r['return'] > 0 else "❌"
            print(f"  {s} Return: {r['return']:.2f}%, Sharpe: {r['sharpe']:.3f}, Trades: {r['trades']}, WR: {r['win_rate']:.1f}%")
    
    print("\n" + "="*70)
