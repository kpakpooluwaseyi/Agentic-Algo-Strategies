"""
Weighted Signals V6 - Hybrid Optimization
==========================================
Combines best elements from V2 (stability) and V5 (selectivity):
- V2's threshold 70 and ATR 2.5x stop
- V5's histogram momentum filter (reduces bad entries)
- NEW: Lower target (4x ATR instead of 5x) for higher hit rate
- NEW: Add EMA alignment bonus for stronger confluence
- NEW: Run full WFA with 70/30 split

Author: Antigravity (Autonomous Research)
Date: 2026-02-10
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


class WeightedSignalsV6(Strategy):
    """
    V6 = V2 stability + V5 selectivity + optimized R:R
    """
    
    # Same weights as V2/V5
    w_macd = 30
    w_rsi = 25
    w_ema = 20
    w_volume = 15
    w_candle = 10
    
    # V2's winning threshold
    entry_threshold = 70
    
    # V2 stop, V2 target (not V5's aggressive 5x)
    atr_period = 14
    atr_stop_mult = 2.5
    atr_target_mult = 4.0  # 1.6:1 R:R (V2's value)
    risk_per_trade = 0.015
    min_move_pct = 0.012  # Slightly lower to increase trades
    
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
        """MACD histogram increasing (from V5)."""
        if len(self.macd_hist) < 2 or np.isnan(self.macd_hist[-1]) or np.isnan(self.macd_hist[-2]):
            return False
        return self.macd_hist[-1] > self.macd_hist[-2]
    
    def hist_momentum_down(self):
        """MACD histogram decreasing (from V5)."""
        if len(self.macd_hist) < 2 or np.isnan(self.macd_hist[-1]) or np.isnan(self.macd_hist[-2]):
            return False
        return self.macd_hist[-1] < self.macd_hist[-2]
    
    def ema_alignment_bullish(self):
        """Check if EMAs are aligned bullishly (20 > 50 > 200)."""
        if any(np.isnan(x) for x in [self.ema_20[-1], self.ema_50[-1], self.ema_200[-1]]):
            return False
        return self.ema_20[-1] > self.ema_50[-1] > self.ema_200[-1]
    
    def ema_alignment_bearish(self):
        """Check if EMAs are aligned bearishly (20 < 50 < 200)."""
        if any(np.isnan(x) for x in [self.ema_20[-1], self.ema_50[-1], self.ema_200[-1]]):
            return False
        return self.ema_20[-1] < self.ema_50[-1] < self.ema_200[-1]
    
    def calculate_long_score(self):
        score = 0
        
        # MACD bullish cross
        if len(self.macd_line) > 1 and not np.isnan(self.macd_line[-1]):
            if self.macd_line[-1] > self.macd_signal[-1] and self.macd_line[-2] <= self.macd_signal[-2]:
                score += self.w_macd
        
        # RSI oversold
        if not np.isnan(self.rsi[-1]) and self.rsi[-1] < 35:
            score += self.w_rsi
        
        # Price above EMA20
        if not np.isnan(self.ema_20[-1]) and self.data.Close[-1] > self.ema_20[-1]:
            score += self.w_ema
        
        # Volume spike
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
        
        # RSI overbought
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
        
        # V6: Require histogram momentum (from V5) OR full EMA alignment
        momentum_long = self.hist_momentum_up() or self.ema_alignment_bullish()
        momentum_short = self.hist_momentum_down() or self.ema_alignment_bearish()
        
        go_long = long_score >= self.entry_threshold and trend >= 0 and momentum_long
        go_short = short_score >= self.entry_threshold and trend <= 0 and momentum_short
        
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


def run_wfa(data_path):
    """Walk-Forward Analysis with 70/30 split."""
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    data = data.dropna()
    
    split_idx = int(len(data) * 0.7)
    
    bt_is = Backtest(data.iloc[:split_idx], WeightedSignalsV6, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats_is = bt_is.run()
    
    bt_oos = Backtest(data.iloc[split_idx:], WeightedSignalsV6, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats_oos = bt_oos.run()
    
    return {
        'is_return': float(stats_is['Return [%]']) if pd.notna(stats_is['Return [%]']) else 0,
        'oos_return': float(stats_oos['Return [%]']) if pd.notna(stats_oos['Return [%]']) else 0,
        'is_trades': int(stats_is['# Trades']),
        'oos_trades': int(stats_oos['# Trades']),
        'is_sharpe': float(stats_is['Sharpe Ratio']) if pd.notna(stats_is['Sharpe Ratio']) else 0,
        'oos_sharpe': float(stats_oos['Sharpe Ratio']) if pd.notna(stats_oos['Sharpe Ratio']) else 0,
        'is_win_rate': float(stats_is['Win Rate [%]']) if pd.notna(stats_is['Win Rate [%]']) else 0,
        'oos_win_rate': float(stats_oos['Win Rate [%]']) if pd.notna(stats_oos['Win Rate [%]']) else 0,
    }


if __name__ == '__main__':
    print("="*70)
    print("WEIGHTED SIGNALS V6 - HYBRID (V2 stability + V5 selectivity)")
    print("="*70)
    
    base_path = "/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/data"
    main_dataset = f"{base_path}/crypto/BTC-USDT_15m_160weeks.csv"
    
    # Full backtest
    print("\n📊 FULL BACKTEST")
    data = pd.read_csv(main_dataset, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, WeightedSignalsV6, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    ret = float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0
    sharpe = float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0
    trades = int(stats['# Trades'])
    wr = float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0
    max_dd = float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0
    
    status = "✅" if ret > 0 else "❌"
    print(f"BTC 15m (2022-2025): {status} Return: {ret:+.2f}%, Sharpe: {sharpe:.3f}, Trades: {trades}, WR: {wr:.1f}%, MaxDD: {max_dd:.1f}%")
    
    # Walk-Forward Analysis
    print("\n📈 WALK-FORWARD ANALYSIS (70/30)")
    wfa = run_wfa(main_dataset)
    print(f"IS:  {wfa['is_return']:+.2f}% ({wfa['is_trades']} trades, Sharpe: {wfa['is_sharpe']:.3f}, WR: {wfa['is_win_rate']:.1f}%)")
    print(f"OOS: {wfa['oos_return']:+.2f}% ({wfa['oos_trades']} trades, Sharpe: {wfa['oos_sharpe']:.3f}, WR: {wfa['oos_win_rate']:.1f}%)")
    
    # Comparison table
    print("\n" + "="*70)
    print("VERSION COMPARISON")
    print("="*70)
    print(f"{'Version':<10} {'Return':>10} {'Sharpe':>10} {'Trades':>10} {'WR':>10}")
    print("-"*50)
    print(f"{'V2':<10} {'+0.60%':>10} {'0.108':>10} {'43':>10} {'53.5%':>10}")
    print(f"{'V5':<10} {'-2.48%':>10} {'-0.358':>10} {'64':>10} {'42.2%':>10}")
    print(f"{'V6':<10} {f'{ret:+.2f}%':>10} {f'{sharpe:.3f}':>10} {str(trades):>10} {f'{wr:.1f}%':>10}")
    
    # Score
    score = 0
    if ret > 0: score += 2
    if trades > 40: score += 2
    elif trades > 25: score += 1
    if sharpe > 0.3: score += 2
    elif sharpe > 0.1: score += 1
    if wfa['oos_return'] > 0: score += 2
    if wr > 50: score += 2
    elif wr > 45: score += 1
    
    print(f"\n📊 SCORE: {score}/10 {'✅ PASSED' if score >= 6 else '⚠️ NEEDS WORK' if score >= 4 else '❌ FAILED'}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output = {
        'full_backtest': {'return': ret, 'sharpe': sharpe, 'trades': trades, 'win_rate': wr, 'max_dd': max_dd},
        'wfa': wfa,
        'score': score,
        'comparison': {
            'v2': {'return': 0.60, 'sharpe': 0.108, 'trades': 43, 'win_rate': 53.5},
            'v5': {'return': -2.48, 'sharpe': -0.358, 'trades': 64, 'win_rate': 42.2},
        }
    }
    with open('results/weighted_signals_v6.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to results/weighted_signals_v6.json")
