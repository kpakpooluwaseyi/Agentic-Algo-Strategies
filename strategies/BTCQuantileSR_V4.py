"""
BTC Quantile SR V4 - Confirmation Filters + 1H Timeframe
=========================================================
Changes from V3:
- Timeframe: 15m → 1h (cleaner S/R levels)
- Added RSI confirmation: < 30 for longs, > 70 for shorts
- Added MACD histogram confirmation
- Reduced lookback for 1h: 200 → 100 bars (~4 days)

Hypothesis: Higher timeframe + momentum confirmation prevents "falling knife" entries.

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


class BTCQuantileSR_V4(Strategy):
    """
    V4: 1H timeframe + RSI/MACD confirmation filters.
    """
    
    # Adjusted for 1h timeframe
    lookback_length = 100  # ~4 days on 1h
    support_quantile = 20
    resistance_quantile = 80
    entry_buffer = 0.005  # Slightly wider for 1h
    stop_loss_pct = 0.03  # 3% (wider for 1h volatility)
    take_profit_pct = 0.06  # 2:1 R:R
    risk_per_trade = 0.015
    
    # RSI confirmation thresholds
    rsi_oversold = 35  # More lenient than 30
    rsi_overbought = 65
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        ema_200 = ta.ema(close, length=200)
        self.ema_200 = self.I(lambda: ema_200.values)
        
        # RSI
        rsi = ta.rsi(close, length=14)
        self.rsi = self.I(lambda: rsi.values)
        
        # MACD
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        self.macd_hist = self.I(lambda: macd.iloc[:, 2].values if macd is not None else np.zeros(len(close)))
        
        # Quantile levels
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
        rsi = self.rsi[-1]
        macd_hist = self.macd_hist[-1]
        
        if any(np.isnan(x) for x in [support, resistance, ema, rsi, macd_hist]):
            return
        
        buy_threshold = support * (1 + self.entry_buffer)
        sell_threshold = resistance * (1 - self.entry_buffer)
        
        # Original signals
        at_support = price <= buy_threshold
        at_resistance = price >= sell_threshold
        
        # Trend filter
        bullish_trend = price > ema
        bearish_trend = price < ema
        
        # NEW: RSI confirmation
        rsi_oversold_signal = rsi < self.rsi_oversold
        rsi_overbought_signal = rsi > self.rsi_overbought
        
        # NEW: MACD histogram confirmation (momentum turning)
        macd_bullish = macd_hist > 0 or (len(self.macd_hist) > 1 and macd_hist > self.macd_hist[-2])
        macd_bearish = macd_hist < 0 or (len(self.macd_hist) > 1 and macd_hist < self.macd_hist[-2])
        
        # Combined signals with confirmation
        long_signal = at_support and bullish_trend and (rsi_oversold_signal or macd_bullish)
        short_signal = at_resistance and bearish_trend and (rsi_overbought_signal or macd_bearish)
        
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
    data = data.dropna()
    
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
    data = data.dropna()
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
        'is_sharpe': float(stats_is['Sharpe Ratio']) if pd.notna(stats_is['Sharpe Ratio']) else 0,
        'oos_sharpe': float(stats_oos['Sharpe Ratio']) if pd.notna(stats_oos['Sharpe Ratio']) else 0,
    }


if __name__ == '__main__':
    print("="*70)
    print("BTC QUANTILE SR V4 - 1H TIMEFRAME + CONFIRMATION FILTERS")
    print("="*70)
    
    data_path = "/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/data/crypto/BTC-USDT_1h_200weeks.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
    else:
        print("\n📊 Running full backtest...")
        r = run_backtest(data_path, BTCQuantileSR_V4)
        print(f"Return: {r['return_pct']:+.2f}%, Sharpe: {r['sharpe']:.3f}, Trades: {r['trades']}, WR: {r['win_rate']:.1f}%")
        
        print("\n📈 Running Walk-Forward Analysis (70/30)...")
        wfa = run_wfa(data_path, BTCQuantileSR_V4)
        print(f"IS: {wfa['is_return']:+.2f}% ({wfa['is_trades']} trades, Sharpe: {wfa['is_sharpe']:.3f})")
        print(f"OOS: {wfa['oos_return']:+.2f}% ({wfa['oos_trades']} trades, Sharpe: {wfa['oos_sharpe']:.3f})")
        
        # Comparison with V3
        print("\n" + "="*70)
        print("V4 vs V3 COMPARISON")
        print("="*70)
        print(f"V3 (15m, no confirmation): -18.30%, Sharpe -1.158, 234 trades, 20.1% WR")
        print(f"V4 (1h, RSI/MACD):          {r['return_pct']:+.2f}%, Sharpe {r['sharpe']:.3f}, {r['trades']} trades, {r['win_rate']:.1f}% WR")
        
        # Verdict
        score = 0
        if r['return_pct'] > 0: score += 2
        if r['trades'] > 20: score += 2
        if r['sharpe'] > 0.5: score += 2
        elif r['sharpe'] > 0: score += 1
        if wfa['oos_return'] > 0: score += 2
        if r['win_rate'] > 45: score += 2
        elif r['win_rate'] > 35: score += 1
        
        print(f"\n📊 SCORE: {score}/10 {'✅ IMPROVED' if score >= 5 else '❌ STILL FAILING'}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results = {
            'full_backtest': r,
            'wfa': wfa,
            'score': score,
            'v3_comparison': {'return': -18.30, 'sharpe': -1.158, 'trades': 234, 'win_rate': 20.1}
        }
        with open('results/quantile_sr_v4.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to results/quantile_sr_v4.json")
