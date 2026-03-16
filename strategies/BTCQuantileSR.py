"""
BTC Quantile SR with Trend Filter Strategy
============================================
Implements quantile-based support/resistance with EMA200 trend filter.

Thesis:
- Support = 20th percentile of lows (200 bars)
- Resistance = 80th percentile of highs (200 bars)
- Long: Price near support AND above EMA200
- Short: Price near resistance AND below EMA200

Author: Agentic Quant Loop (Developer Phase)
Version: 1.1
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


class BTCQuantileSR(Strategy):
    """
    BTC Quantile Support/Resistance Strategy
    
    Parameters:
    - lookback_length: Window for percentile calculation
    - support_quantile: Percentile for support (0-100)
    - resistance_quantile: Percentile for resistance (0-100)
    - entry_buffer: Buffer around levels
    - stop_loss_pct: Stop loss percentage
    """
    
    lookback_length = 200
    support_quantile = 20
    resistance_quantile = 80
    entry_buffer = 0.003  # 0.3%
    stop_loss_pct = 0.015  # 1.5%
    risk_per_trade = 0.015
    
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
    
    def next(self):
        if len(self.data) < 210:
            return
        
        price = self.data.Close[-1]
        support = self.support[-1]
        resistance = self.resistance[-1]
        ema = self.ema_200[-1]
        
        if np.isnan(support) or np.isnan(resistance) or np.isnan(ema):
            return
        
        # Calculate thresholds with buffer
        buy_threshold = support * (1 + self.entry_buffer)
        sell_threshold = resistance * (1 - self.entry_buffer)
        
        # Check for entry signals
        long_signal = price <= buy_threshold and price > ema
        short_signal = price >= sell_threshold and price < ema
        
        # Dynamic flipping: close opposite position on new signal
        if self.position.is_short and long_signal:
            self.position.close()
        elif self.position.is_long and short_signal:
            self.position.close()
        
        # Entry logic
        if not self.position:
            if long_signal:
                # Calculate position size
                risk_amount = self.equity * self.risk_per_trade
                stop_distance = price * self.stop_loss_pct
                shares = max(1, min(int(risk_amount / stop_distance), 
                                    int(self.equity * 0.3 / price)))
                
                sl = price * (1 - self.stop_loss_pct)
                self.buy(size=shares, sl=sl)
            
            elif short_signal:
                risk_amount = self.equity * self.risk_per_trade
                stop_distance = price * self.stop_loss_pct
                shares = max(1, min(int(risk_amount / stop_distance), 
                                    int(self.equity * 0.3 / price)))
                
                sl = price * (1 + self.stop_loss_pct)
                self.sell(size=shares, sl=sl)


def run_backtest(data_path: str, strategy_class, name: str = ""):
    """Run backtest and return stats."""
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
        'profit_factor': float(stats['Profit Factor']) if pd.notna(stats.get('Profit Factor', np.nan)) else 0,
    }


def run_wfa(data_path: str, strategy_class, train_pct: float = 0.7):
    """Walk-Forward Analysis: 70% IS, 30% OOS."""
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    split_idx = int(len(data) * train_pct)
    is_data = data.iloc[:split_idx]
    oos_data = data.iloc[split_idx:]
    
    # In-sample
    bt_is = Backtest(is_data, strategy_class, cash=1_000_000, 
                    commission=0.002, trade_on_close=True)
    stats_is = bt_is.run()
    
    # Out-of-sample
    bt_oos = Backtest(oos_data, strategy_class, cash=1_000_000, 
                     commission=0.002, trade_on_close=True)
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
    print("BTC QUANTILE SR WITH TREND FILTER - BACKTEST")
    print("="*70)
    
    data_path = "data/crypto/BTC-USDT_15m_160weeks.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        exit(1)
    
    # Full backtest
    print("\n📊 FULL BACKTEST")
    print("-" * 50)
    results = run_backtest(data_path, BTCQuantileSR, "BTC 15m Full")
    
    print(f"Return:        {results['return_pct']:+.2f}%")
    print(f"Sharpe:        {results['sharpe']:.3f}")
    print(f"Max Drawdown:  {results['max_dd']:.2f}%")
    print(f"Trade Count:   {results['trades']}")
    print(f"Win Rate:      {results['win_rate']:.1f}%")
    
    # Walk-Forward Analysis
    print("\n📈 WALK-FORWARD ANALYSIS (70/30 Split)")
    print("-" * 50)
    wfa = run_wfa(data_path, BTCQuantileSR)
    
    print(f"In-Sample:     Return={wfa['is_return']:+.2f}%, Sharpe={wfa['is_sharpe']:.3f}, Trades={wfa['is_trades']}")
    print(f"Out-of-Sample: Return={wfa['oos_return']:+.2f}%, Sharpe={wfa['oos_sharpe']:.3f}, Trades={wfa['oos_trades']}")
    
    degradation = wfa['is_return'] - wfa['oos_return']
    print(f"Degradation:   {degradation:.2f}%")
    
    # Verdict
    print("\n" + "="*70)
    print("AUDIT VERDICT")
    print("="*70)
    
    score = 0
    checks = []
    
    # Check 1: Positive expectancy
    if results['return_pct'] > 0:
        score += 2
        checks.append("✅ Positive expectancy")
    else:
        checks.append("❌ Negative expectancy")
    
    # Check 2: Trade count > 30
    if results['trades'] > 30:
        score += 2
        checks.append(f"✅ Trade count ({results['trades']} > 30)")
    else:
        checks.append(f"❌ Low trade count ({results['trades']})")
    
    # Check 3: Sharpe > 1.0
    if results['sharpe'] > 1.0:
        score += 2
        checks.append(f"✅ Sharpe > 1.0 ({results['sharpe']:.3f})")
    elif results['sharpe'] > 0.5:
        score += 1
        checks.append(f"⚠️ Sharpe marginal ({results['sharpe']:.3f})")
    else:
        checks.append(f"❌ Sharpe < 0.5 ({results['sharpe']:.3f})")
    
    # Check 4: OOS performance positive
    if wfa['oos_return'] > 0:
        score += 2
        checks.append(f"✅ OOS positive ({wfa['oos_return']:+.2f}%)")
    else:
        checks.append(f"❌ OOS negative ({wfa['oos_return']:+.2f}%)")
    
    # Check 5: Win rate > 40%
    if results['win_rate'] > 50:
        score += 2
        checks.append(f"✅ Win rate ({results['win_rate']:.1f}% > 50%)")
    elif results['win_rate'] > 40:
        score += 1
        checks.append(f"⚠️ Win rate marginal ({results['win_rate']:.1f}%)")
    else:
        checks.append(f"❌ Win rate low ({results['win_rate']:.1f}%)")
    
    for check in checks:
        print(check)
    
    print(f"\n📊 FINAL SCORE: {score}/10")
    
    if score >= 8:
        print("🏆 STATUS: PASSED")
    else:
        print("❌ STATUS: FAILED - Needs iteration")
    
    print("="*70)
