"""
Quantile S/R Multi-Asset Test
==============================
Tests the Quantile Support/Resistance hypothesis on non-crypto assets
to determine if the mean-reversion concept works better on less volatile markets.

Assets: SPY, QQQ, GLD, EURUSD

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


def rolling_percentile(arr, window, percentile):
    result = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        result[i] = np.percentile(arr[i-window:i], percentile)
    return result


class QuantileSRMultiAsset(Strategy):
    """
    Quantile-based support/resistance with EMA trend filter.
    Adapted from BTCQuantileSR_V3 for multi-asset testing.
    """
    
    lookback_length = 200
    support_quantile = 20
    resistance_quantile = 80
    entry_buffer = 0.003
    stop_loss_pct = 0.025
    take_profit_pct = 0.05  # 2:1 R:R
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


def run_backtest(data_path, name):
    """Run backtest on a single asset."""
    try:
        data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
        data.columns = [c.strip().capitalize() for c in data.columns]
        
        # Clean data
        data = data.dropna()
        if len(data) < 250:
            return {'name': name, 'error': 'Insufficient data'}
        
        bt = Backtest(data, QuantileSRMultiAsset, cash=100_000, 
                     commission=0.001, trade_on_close=True)
        stats = bt.run()
        
        return {
            'name': name,
            'return_pct': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
            'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
            'max_dd': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
            'trades': int(stats['# Trades']),
            'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        }
    except Exception as e:
        return {'name': name, 'error': str(e)}


if __name__ == '__main__':
    print("="*70)
    print("QUANTILE S/R MULTI-ASSET TEST")
    print("Testing hypothesis on non-crypto assets")
    print("="*70)
    
    base_path = "/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/data"
    
    datasets = [
        (f"{base_path}/equities/SPY_15m.csv", "SPY (S&P 500 ETF)"),
        (f"{base_path}/equities/QQQ_15m.csv", "QQQ (Nasdaq 100 ETF)"),
        (f"{base_path}/commodities/GLD_15m.csv", "GLD (Gold ETF)"),
        (f"{base_path}/forex/EURUSD=X_15m.csv", "EURUSD (Forex)"),
        (f"{base_path}/equities/NVDA_15m.csv", "NVDA (Nvidia)"),
        (f"{base_path}/equities/TSLA_15m.csv", "TSLA (Tesla)"),
    ]
    
    results = []
    
    for path, name in datasets:
        if os.path.exists(path):
            print(f"\n📊 Testing {name}...")
            r = run_backtest(path, name)
            results.append(r)
            
            if 'error' in r:
                print(f"  ❌ Error: {r['error']}")
            else:
                status = "✅" if r['return_pct'] > 0 else "❌"
                print(f"  {status} Return: {r['return_pct']:+.2f}%, Sharpe: {r['sharpe']:.3f}, "
                      f"Trades: {r['trades']}, WR: {r['win_rate']:.1f}%")
        else:
            print(f"\n⚠️ File not found: {path}")
    
    # Summary
    print("\n" + "="*70)
    print("MULTI-ASSET SUMMARY")
    print("="*70)
    
    valid_results = [r for r in results if 'error' not in r]
    profitable = [r for r in valid_results if r['return_pct'] > 0]
    
    print(f"Profitable: {len(profitable)}/{len(valid_results)}")
    
    if valid_results:
        avg_return = np.mean([r['return_pct'] for r in valid_results])
        avg_sharpe = np.mean([r['sharpe'] for r in valid_results])
        print(f"Avg Return: {avg_return:+.2f}%")
        print(f"Avg Sharpe: {avg_sharpe:.3f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/quantile_sr_multi_asset.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to results/quantile_sr_multi_asset.json")
    
    # Verdict
    if len(profitable) >= len(valid_results) / 2:
        print("\n🎯 VERDICT: Quantile S/R shows promise on non-crypto assets!")
    else:
        print("\n❌ VERDICT: Quantile S/R does not generalize to non-crypto assets.")
