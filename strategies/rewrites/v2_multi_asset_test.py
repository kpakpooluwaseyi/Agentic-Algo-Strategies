"""
Multi-Asset Test for Optimized V2
==================================
Test the optimized V2 (thresh=65, stop=2.5x, target=3.0x) across multiple 15m assets.

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json
import warnings
warnings.filterwarnings('ignore')


def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    return pd.Series(tr).rolling(period).mean().values


class WeightedSignalsOptimized(Strategy):
    """Optimized V2 with best parameters."""
    
    w_macd = 30
    w_rsi = 25
    w_ema = 20
    w_volume = 15
    w_candle = 10
    entry_threshold = 65  # Optimized
    atr_stop_mult = 2.5   # Optimized
    atr_target_mult = 3.0 # Optimized
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


def test_asset(path, name):
    """Test on a single asset."""
    try:
        data = pd.read_csv(path, parse_dates=[0], index_col=0)
        data.columns = [c.strip().capitalize() for c in data.columns]
        
        bt = Backtest(data, WeightedSignalsOptimized, cash=1_000_000, commission=0.002, trade_on_close=True)
        stats = bt.run()
        
        return {
            'asset': name,
            'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
            'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
            'trades': int(stats['# Trades']),
            'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        }
    except Exception as e:
        return {'asset': name, 'error': str(e)[:50]}


if __name__ == '__main__':
    print("="*70)
    print("MULTI-ASSET TEST - OPTIMIZED WEIGHTED SIGNALS V2")
    print("Parameters: thresh=65, stop=2.5x, target=3.0x")
    print("="*70)
    
    # 15m datasets
    datasets = [
        ("data/crypto/BTC-USDT_15m_160weeks.csv", "BTC 15m"),
        ("data/crypto/BTCUSDT_P_15m_2025.csv", "BTC 15m 2025"),
        ("data/crypto/ETHUSD_15m.csv", "ETH 15m"),
        ("data/equities/SPY_15m.csv", "SPY 15m"),
        ("data/equities/QQQ_15m.csv", "QQQ 15m"),
        ("data/forex/EURUSD_15m.csv", "EURUSD 15m"),
        ("data/commodities/GLD_15m.csv", "GLD 15m"),
    ]
    
    # Also test 1h datasets
    datasets_1h = [
        ("data/crypto/BTC-USDT_1h_200weeks.csv", "BTC 1h"),
        ("data/equities/SPY_1h.csv", "SPY 1h"),
    ]
    
    all_datasets = datasets + datasets_1h
    
    results = []
    
    for path, name in all_datasets:
        if os.path.exists(path):
            print(f"\nTesting {name}...", end=" ")
            r = test_asset(path, name)
            results.append(r)
            
            if 'error' in r:
                print(f"❌ Error: {r['error']}")
            else:
                status = "✅" if r['return'] > 0 else "❌"
                print(f"{status} Return: {r['return']:.2f}%, Sharpe: {r['sharpe']:.3f}, Trades: {r['trades']}")
        else:
            print(f"\n⚠️ {name}: File not found")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    valid = [r for r in results if 'return' in r]
    profitable = [r for r in valid if r['return'] > 0]
    
    print(f"\nTested: {len(valid)} assets")
    print(f"Profitable: {len(profitable)}/{len(valid)}")
    
    if profitable:
        print("\n✅ Profitable Assets:")
        for r in sorted(profitable, key=lambda x: -x['return']):
            print(f"   {r['asset']}: +{r['return']:.2f}%, Sharpe={r['sharpe']:.3f}")
    
    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/multi_asset_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/multi_asset_results.json")
