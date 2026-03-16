"""
BTC Quantile SR with Trend Filter - V2 (Iteration 1)
======================================================
Fixes from Cycle 1:
1. Added take profit (2x stop loss)
2. Widened stop loss (2.5% ATR-based)
3. Tightened entry buffer (0.1%)
4. Added ATR filter (skip low vol)

Author: Agentic Quant Loop (Iteration 1)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import warnings
warnings.filterwarnings('ignore')


def rolling_percentile(arr, window, percentile):
    """Calculate rolling percentile over window."""
    result = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        result[i] = np.percentile(arr[i-window:i], percentile)
    return result


def calculate_atr(high, low, close, period=14):
    """Calculate ATR."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    return pd.Series(tr).rolling(period).mean().values


class BTCQuantileSR_V2(Strategy):
    """
    V2 Fixes:
    - Take profit added (2x stop)
    - ATR-based stops
    - Tighter entry buffer
    - Volatility filter
    """
    
    lookback_length = 200
    support_quantile = 20
    resistance_quantile = 80
    entry_buffer = 0.001  # Tightened from 0.003
    atr_stop_mult = 2.0   # ATR-based stop
    atr_target_mult = 4.0 # 2:1 R:R
    risk_per_trade = 0.015
    min_atr_pct = 0.005   # Skip if ATR < 0.5%
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # EMA200 for trend filter
        ema_200 = ta.ema(close, length=200)
        self.ema_200 = self.I(lambda: ema_200.values)
        
        # Rolling percentiles for support/resistance
        self.support = self.I(rolling_percentile, 
                              low.values, 
                              self.lookback_length, 
                              self.support_quantile)
        
        self.resistance = self.I(rolling_percentile, 
                                 high.values, 
                                 self.lookback_length, 
                                 self.resistance_quantile)
        
        # ATR for volatility filter and stops
        self.atr = self.I(calculate_atr, high.values, low.values, close.values, 14)
    
    def next(self):
        if len(self.data) < 210:
            return
        
        price = self.data.Close[-1]
        support = self.support[-1]
        resistance = self.resistance[-1]
        ema = self.ema_200[-1]
        atr = self.atr[-1]
        
        if np.isnan(support) or np.isnan(resistance) or np.isnan(ema) or np.isnan(atr):
            return
        
        # Volatility filter: skip low vol
        if atr / price < self.min_atr_pct:
            return
        
        # Calculate thresholds with tight buffer
        buy_threshold = support * (1 + self.entry_buffer)
        sell_threshold = resistance * (1 - self.entry_buffer)
        
        # Check for entry signals
        long_signal = price <= buy_threshold and price > ema
        short_signal = price >= sell_threshold and price < ema
        
        # Dynamic flipping still allowed but we also have TP now
        if self.position.is_short and long_signal:
            self.position.close()
        elif self.position.is_long and short_signal:
            self.position.close()
        
        # Entry logic with ATR-based stops and targets
        if not self.position:
            stop_distance = atr * self.atr_stop_mult
            target_distance = atr * self.atr_target_mult
            
            risk_amount = self.equity * self.risk_per_trade
            shares = max(1, min(int(risk_amount / stop_distance), 
                                int(self.equity * 0.3 / price)))
            
            if long_signal:
                sl = price - stop_distance
                tp = price + target_distance
                self.buy(size=shares, sl=sl, tp=tp)
            
            elif short_signal:
                sl = price + stop_distance
                tp = price - target_distance
                self.sell(size=shares, sl=sl, tp=tp)


def run_backtest(data_path, strategy_class, name=""):
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, strategy_class, cash=1_000_000, 
                 commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    return {
        'name': name,
        'return_pct': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_dd': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'trades': int(stats['# Trades']),
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
    }


def run_wfa(data_path, strategy_class, train_pct=0.7):
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    split_idx = int(len(data) * train_pct)
    is_data = data.iloc[:split_idx]
    oos_data = data.iloc[split_idx:]
    
    bt_is = Backtest(is_data, strategy_class, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats_is = bt_is.run()
    
    bt_oos = Backtest(oos_data, strategy_class, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats_oos = bt_oos.run()
    
    return {
        'is_return': float(stats_is['Return [%]']) if pd.notna(stats_is['Return [%]']) else 0,
        'is_sharpe': float(stats_is['Sharpe Ratio']) if pd.notna(stats_is['Sharpe Ratio']) else 0,
        'is_trades': int(stats_is['# Trades']),
        'oos_return': float(stats_oos['Return [%]']) if pd.notna(stats_oos['Return [%]']) else 0,
        'oos_sharpe': float(stats_oos['Sharpe Ratio']) if pd.notna(stats_oos['Sharpe Ratio']) else 0,
        'oos_trades': int(stats_oos['# Trades']),
    }


if __name__ == '__main__':
    print("="*70)
    print("BTC QUANTILE SR V2 - ITERATION 1")
    print("Fixes: TP added, ATR stops, tight buffer, vol filter")
    print("="*70)
    
    data_path = "data/crypto/BTC-USDT_15m_160weeks.csv"
    
    print("\n📊 FULL BACKTEST")
    print("-" * 50)
    r = run_backtest(data_path, BTCQuantileSR_V2, "BTC 15m Full")
    
    print(f"Return:        {r['return_pct']:+.2f}%")
    print(f"Sharpe:        {r['sharpe']:.3f}")
    print(f"Max Drawdown:  {r['max_dd']:.2f}%")
    print(f"Trade Count:   {r['trades']}")
    print(f"Win Rate:      {r['win_rate']:.1f}%")
    
    print("\n📈 WALK-FORWARD ANALYSIS")
    print("-" * 50)
    wfa = run_wfa(data_path, BTCQuantileSR_V2)
    print(f"IS:  Return={wfa['is_return']:+.2f}%, Sharpe={wfa['is_sharpe']:.3f}, Trades={wfa['is_trades']}")
    print(f"OOS: Return={wfa['oos_return']:+.2f}%, Sharpe={wfa['oos_sharpe']:.3f}, Trades={wfa['oos_trades']}")
    
    print("\n" + "="*70)
    print("AUDIT SCORING")
    print("="*70)
    
    score = 0
    if r['return_pct'] > 0: score += 2
    if r['trades'] > 30: score += 2
    if r['sharpe'] > 1.0: score += 2
    elif r['sharpe'] > 0.5: score += 1
    if wfa['oos_return'] > 0: score += 2
    if r['win_rate'] > 50: score += 2
    elif r['win_rate'] > 40: score += 1
    
    print(f"\n📊 FINAL SCORE: {score}/10")
    print("🏆 PASSED" if score >= 8 else "❌ FAILED - Needs iteration")
    print("="*70)
