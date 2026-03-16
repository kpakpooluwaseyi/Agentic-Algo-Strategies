"""
Multi-Asset Strategy Tester
============================
Tests regime-filtered strategies across multiple assets.
This helps identify strategies that work across asset classes.

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta as ta
import os
import json
from pathlib import Path


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(period).mean().values
    return atr


class SimpleBreakoutStrategy(Strategy):
    """
    Simple breakout with trend and volatility regime filter.
    Designed to work across multiple asset classes.
    
    Regime: Only trade when:
    - ADX > 25 (strong trend)
    - Price trending with 50 EMA slope
    
    Entry: Donchian 20-bar breakout in trend direction
    Exit: Trailing stop at 2x ATR
    """
    
    channel_period = 20
    ema_period = 50
    adx_period = 14
    adx_threshold = 25
    atr_period = 14
    atr_trail_mult = 2.0
    risk_per_trade = 0.01
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Donchian Channels
        self.channel_high = self.I(lambda: high.shift(1).rolling(self.channel_period).max().values)
        self.channel_low = self.I(lambda: low.shift(1).rolling(self.channel_period).min().values)
        
        # EMA and its slope
        ema = ta.ema(close, length=self.ema_period)
        self.ema = self.I(lambda: ema.values)
        
        # ADX
        adx = ta.adx(high, low, close, length=self.adx_period)
        self.adx = self.I(lambda: adx.iloc[:, 0].values)
        
        # ATR
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close,
                         self.atr_period)
        
        self.trailing_stop = None
    
    def next(self):
        if len(self.data) < 60:
            return
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        adx = self.adx[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        # Manage position with trailing stop
        if self.position:
            if self.position.is_long:
                new_stop = price - (atr * self.atr_trail_mult)
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                if price < self.trailing_stop:
                    self.position.close()
                    self.trailing_stop = None
            elif self.position.is_short:
                new_stop = price + (atr * self.atr_trail_mult)
                if self.trailing_stop is None or new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                if price > self.trailing_stop:
                    self.position.close()
                    self.trailing_stop = None
            return
        
        # Regime: ADX strong trend + EMA slope
        ema_slope_up = self.ema[-1] > self.ema[-5] if len(self.ema) > 5 else False
        ema_slope_down = self.ema[-1] < self.ema[-5] if len(self.ema) > 5 else False
        strong_trend = not np.isnan(adx) and adx > self.adx_threshold
        
        uptrend = ema_slope_up and strong_trend and price > self.ema[-1]
        downtrend = ema_slope_down and strong_trend and price < self.ema[-1]
        
        # Breakout signals
        break_high = high > self.channel_high[-1]
        break_low = low < self.channel_low[-1]
        
        # Position sizing
        risk_amount = self.equity * self.risk_per_trade
        sl_distance = atr * self.atr_trail_mult
        
        if sl_distance <= 0:
            return
        
        position_size = risk_amount / sl_distance
        max_shares = int(self.equity * 0.5 / price)
        shares = min(int(position_size), max_shares)
        
        if shares < 1:
            shares = 1
        
        # Long in uptrend
        if break_high and uptrend:
            self.trailing_stop = price - sl_distance
            self.buy(size=shares, sl=self.trailing_stop)
        
        # Short in downtrend
        elif break_low and downtrend:
            self.trailing_stop = price + sl_distance
            self.sell(size=shares, sl=self.trailing_stop)


def run_multi_asset_test():
    """Test strategy across multiple assets."""
    
    datasets = [
        ("BTC 1H 2021-2025", "data/crypto/BTC-USDT_1h_200weeks.csv"),
        ("BTC 15m 2025", "data/crypto/BTCUSDT_P_15m_2025.csv"),
        ("ETH 15m", "data/crypto/ETHUSD_15m.csv"),
        ("SPY 15m", "data/equities/SPY_15m.csv"),
        ("QQQ 15m", "data/equities/QQQ_15m.csv"),
        ("GLD 15m", "data/commodities/GLD_15m.csv"),
    ]
    
    results = []
    
    print("="*70)
    print("MULTI-ASSET STRATEGY TEST")
    print("="*70)
    
    for name, path in datasets:
        if not os.path.exists(path):
            print(f"  ❌ {name}: File not found")
            continue
        
        try:
            data = pd.read_csv(path, parse_dates=[0], index_col=0)
            data.columns = [c.strip().capitalize() for c in data.columns]
            
            bt = Backtest(data, SimpleBreakoutStrategy, cash=1_000_000, 
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
            
            sharpe_str = f"{result['sharpe']:.2f}" if result['sharpe'] != 0 else "N/A"
            print(f"  {name}: Return={result['return']:.1f}%, Sharpe={sharpe_str}, Trades={result['trades']}, WR={result['win_rate']:.1f}%")
            
        except Exception as e:
            print(f"  ❌ {name}: {str(e)[:50]}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        avg_return = np.mean([r['return'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results if r['sharpe'] != 0])
        profitabel = sum(1 for r in results if r['return'] > 0)
        
        print(f"Assets Tested:    {len(results)}")
        print(f"Profitable:       {profitabel}/{len(results)}")
        print(f"Avg Return:       {avg_return:.2f}%")
        print(f"Avg Sharpe:       {avg_sharpe:.3f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/multi_asset_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/multi_asset_test_results.json")
    
    return results


if __name__ == '__main__':
    run_multi_asset_test()
