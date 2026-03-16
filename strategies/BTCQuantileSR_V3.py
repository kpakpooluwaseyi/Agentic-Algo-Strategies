"""
BTC Quantile SR V3 - Iteration 2
=================================
V2 over-filtered (only 3 trades). 
V3 keeps V1's entry frequency but adds TP.

Changes from V1:
- KEEP original entry buffer (0.003)  
- ADD take profit = 2x stop
- WIDEN stop loss to 2.5% (was 1.5%)
- ADD ATR proportional sizing

Author: Agentic Quant Loop (Iteration 2)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import warnings
warnings.filterwarnings('ignore')


def rolling_percentile(arr, window, percentile):
    result = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        result[i] = np.percentile(arr[i-window:i], percentile)
    return result


class BTCQuantileSR_V3(Strategy):
    """
    V3: Keep V1 entry frequency, add proper exits.
    """
    
    lookback_length = 200
    support_quantile = 20
    resistance_quantile = 80
    entry_buffer = 0.003  # KEPT from V1 (V2's 0.001 killed trades)
    stop_loss_pct = 0.025  # 2.5% (widened from V1's 1.5%)
    take_profit_pct = 0.05 # 5% (2:1 R:R with 2.5% stop)
    risk_per_trade = 0.015
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        ema_200 = ta.ema(close, length=200)
        self.ema_200 = self.I(lambda: ema_200.values)
        
        self.support = self.I(rolling_percentile, 
                              low.values, 
                              self.lookback_length, 
                              self.support_quantile)
        
        self.resistance = self.I(rolling_percentile, 
                                 high.values, 
                                 self.lookback_length, 
                                 self.resistance_quantile)
    
    def next(self):
        if len(self.data) < 210:
            return
        
        price = self.data.Close[-1]
        support = self.support[-1]
        resistance = self.resistance[-1]
        ema = self.ema_200[-1]
        
        if np.isnan(support) or np.isnan(resistance) or np.isnan(ema):
            return
        
        buy_threshold = support * (1 + self.entry_buffer)
        sell_threshold = resistance * (1 - self.entry_buffer)
        
        long_signal = price <= buy_threshold and price > ema
        short_signal = price >= sell_threshold and price < ema
        
        # Dynamic flipping
        if self.position.is_short and long_signal:
            self.position.close()
        elif self.position.is_long and short_signal:
            self.position.close()
        
        if not self.position:
            stop_distance = price * self.stop_loss_pct
            risk_amount = self.equity * self.risk_per_trade
            shares = max(1, min(int(risk_amount / stop_distance), 
                                int(self.equity * 0.3 / price)))
            
            if long_signal:
                sl = price * (1 - self.stop_loss_pct)
                tp = price * (1 + self.take_profit_pct)
                self.buy(size=shares, sl=sl, tp=tp)
            
            elif short_signal:
                sl = price * (1 + self.stop_loss_pct)
                tp = price * (1 - self.take_profit_pct)
                self.sell(size=shares, sl=sl, tp=tp)


def run_backtest(data_path, strategy_class, name=""):
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    bt = Backtest(data, strategy_class, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats = bt.run()
    return {
        'return_pct': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_dd': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'trades': int(stats['# Trades']),
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
    }


def run_wfa(data_path, strategy_class):
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    split_idx = int(len(data) * 0.7)
    
    bt_is = Backtest(data.iloc[:split_idx], strategy_class, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats_is = bt_is.run()
    
    bt_oos = Backtest(data.iloc[split_idx:], strategy_class, cash=1_000_000, commission=0.002, trade_on_close=True)
    stats_oos = bt_oos.run()
    
    return {
        'is_return': float(stats_is['Return [%]']) if pd.notna(stats_is['Return [%]']) else 0,
        'oos_return': float(stats_oos['Return [%]']) if pd.notna(stats_oos['Return [%]']) else 0,
        'is_trades': int(stats_is['# Trades']),
        'oos_trades': int(stats_oos['# Trades']),
    }


if __name__ == '__main__':
    print("="*70)
    print("BTC QUANTILE SR V3 - ITERATION 2")
    print("Fix: Keep V1 entries, add TP + wider stop")
    print("="*70)
    
    data_path = "data/crypto/BTC-USDT_15m_160weeks.csv"
    
    r = run_backtest(data_path, BTCQuantileSR_V3)
    print(f"\n📊 Return: {r['return_pct']:+.2f}%, Sharpe: {r['sharpe']:.3f}, Trades: {r['trades']}, WR: {r['win_rate']:.1f}%")
    
    wfa = run_wfa(data_path, BTCQuantileSR_V3)
    print(f"📈 IS: {wfa['is_return']:+.2f}% ({wfa['is_trades']} trades) | OOS: {wfa['oos_return']:+.2f}% ({wfa['oos_trades']} trades)")
    
    score = 0
    if r['return_pct'] > 0: score += 2
    if r['trades'] > 30: score += 2
    if r['sharpe'] > 1.0: score += 2
    elif r['sharpe'] > 0.5: score += 1
    if wfa['oos_return'] > 0: score += 2
    if r['win_rate'] > 50: score += 2
    elif r['win_rate'] > 40: score += 1
    
    print(f"\n📊 SCORE: {score}/10 {'✅ PASSED' if score >= 8 else '❌ FAILED'}")
