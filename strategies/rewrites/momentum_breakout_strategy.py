"""
Momentum Breakout Strategy - Alternative Approach
===================================================
Different paradigm from weighted signals:
- Pure momentum/breakout, not mean reversion
- Donchian channel breakout
- Trend confirmation with ADX
- ATR-based risk management

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


class MomentumBreakoutStrategy(Strategy):
    """
    Momentum Breakout Strategy
    
    Entry:
    - LONG: Price breaks above 20-bar high + ADX > 20 + price above EMA50
    - SHORT: Price breaks below 20-bar low + ADX > 20 + price below EMA50
    
    Exit:
    - Stop: 2x ATR
    - Target: 4x ATR (2:1 R:R)
    
    Commission protection:
    - Skip if ATR < 0.5% of price
    """
    
    breakout_period = 20
    adx_period = 14
    adx_threshold = 20
    ema_period = 50
    atr_stop_mult = 2.0
    atr_target_mult = 4.0
    risk_per_trade = 0.015
    min_atr_pct = 0.005  # 0.5% minimum ATR
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Donchian channel (breakout levels)
        self.donchian_high = self.I(lambda: high.rolling(self.breakout_period).max().shift(1).values)
        self.donchian_low = self.I(lambda: low.rolling(self.breakout_period).min().shift(1).values)
        
        # ADX for trend strength
        adx = ta.adx(high, low, close, length=self.adx_period)
        self.adx = self.I(lambda: adx.iloc[:, 0].values)
        
        # EMA for trend direction
        ema = ta.ema(close, length=self.ema_period)
        self.ema = self.I(lambda: ema.values)
        
        # ATR
        self.atr = self.I(calculate_atr, high, low, close, 14)
    
    def next(self):
        if self.position or len(self.data) < 60:
            return
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        # Commission protection
        if atr / price < self.min_atr_pct:
            return
        
        donchian_h = self.donchian_high[-1]
        donchian_l = self.donchian_low[-1]
        adx = self.adx[-1]
        ema = self.ema[-1]
        
        if np.isnan(donchian_h) or np.isnan(adx) or np.isnan(ema):
            return
        
        # Trend strength check
        strong_trend = adx > self.adx_threshold
        
        # Breakout detection
        breakout_up = high > donchian_h and price > ema
        breakout_down = low < donchian_l and price < ema
        
        stop_distance = atr * self.atr_stop_mult
        target_distance = atr * self.atr_target_mult
        
        risk_amount = self.equity * self.risk_per_trade
        shares = max(1, min(int(risk_amount / stop_distance), int(self.equity * 0.3 / price)))
        
        if breakout_up and strong_trend:
            self.buy(size=shares, sl=price - stop_distance, tp=price + target_distance)
        
        elif breakout_down and strong_trend:
            self.sell(size=shares, sl=price + stop_distance, tp=price - target_distance)


def test_asset(path, name):
    try:
        data = pd.read_csv(path, parse_dates=[0], index_col=0)
        data.columns = [c.strip().capitalize() for c in data.columns]
        
        bt = Backtest(data, MomentumBreakoutStrategy, cash=1_000_000, commission=0.002, trade_on_close=True)
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
    print("MOMENTUM BREAKOUT STRATEGY - ALTERNATIVE APPROACH")
    print("Entry: Donchian breakout + ADX + EMA trend")
    print("="*70)
    
    datasets = [
        ("data/crypto/BTC-USDT_15m_160weeks.csv", "BTC 15m"),
        ("data/crypto/BTCUSDT_P_15m_2025.csv", "BTC 15m 2025"),
        ("data/crypto/ETHUSD_15m.csv", "ETH 15m"),
        ("data/crypto/BTC-USDT_1h_200weeks.csv", "BTC 1h"),
    ]
    
    results = []
    
    for path, name in datasets:
        if os.path.exists(path):
            print(f"\nTesting {name}...", end=" ")
            r = test_asset(path, name)
            results.append(r)
            
            if 'error' in r:
                print(f"❌ Error")
            else:
                status = "✅" if r['return'] > 0 else "❌"
                print(f"{status} Return: {r['return']:.2f}%, Sharpe: {r['sharpe']:.3f}, Trades: {r['trades']}, WR: {r['win_rate']:.1f}%")
    
    print("\n" + "="*70)
    print("COMPARISON: Momentum Breakout vs Weighted Signals V2")
    print("="*70)
    
    # Load V2 results for comparison
    v2_btc = {'return': 2.62, 'sharpe': 0.385}
    mb_btc = next((r for r in results if r['asset'] == 'BTC 15m'), None)
    
    if mb_btc and 'return' in mb_btc:
        print(f"\nBTC 15m:")
        print(f"  Weighted Signals V2: Return={v2_btc['return']:.2f}%, Sharpe={v2_btc['sharpe']:.3f}")
        print(f"  Momentum Breakout:   Return={mb_btc['return']:.2f}%, Sharpe={mb_btc['sharpe']:.3f}")
        
        if mb_btc['return'] > v2_btc['return']:
            print("\n✅ Momentum Breakout WINS for BTC 15m")
        else:
            print("\n❌ Weighted Signals V2 still better for BTC 15m")
    
    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/momentum_breakout_results.json', 'w') as f:
        json.dump(results, f, indent=2)
