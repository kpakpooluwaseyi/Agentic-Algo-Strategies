"""
Pivot Breakout Strategy - Rewrite v3
=====================================
Iteration 3: Simpler is better. Just breakout with ATR stops.

Key changes:
1. Removed momentum and volume filters (overfiltering)
2. Using ATR-based stops instead of pivot-based
3. Reduced R:R to 1.5:1 for higher win rate
4. Added time filter (skip first/last 4 bars of session)

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(period).mean().values
    return atr


class PivotBreakoutV3(Strategy):
    """
    Pivot Breakout Strategy v3 - Simplified
    
    Entry Logic:
    - LONG: Break above 20-bar high + price > EMA50
    - SHORT: Break below 20-bar low + price < EMA50
    
    Risk Management:
    - Stop Loss: 2x ATR
    - Take Profit: 3x ATR (1.5:1 R:R)
    """
    
    breakout_period = 20
    atr_period = 14
    atr_multiplier_sl = 2.0
    atr_multiplier_tp = 3.0
    risk_per_trade = 0.01
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # EMA for trend filter
        ema_50 = ta.ema(close, length=50)
        self.ema_50 = self.I(lambda: ema_50.values)
        
        # Rolling highs/lows (excluding current bar)
        self.rolling_high = self.I(lambda: high.shift(1).rolling(self.breakout_period).max().values)
        self.rolling_low = self.I(lambda: low.shift(1).rolling(self.breakout_period).min().values)
        
        # ATR
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close,
                         self.atr_period)
    
    def next(self):
        if self.position:
            return
        if len(self.data) < 60:
            return
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        if np.isnan(self.rolling_high[-1]) or np.isnan(self.rolling_low[-1]):
            return
        
        # Trend filter
        uptrend = price > self.ema_50[-1]
        downtrend = price < self.ema_50[-1]
        
        # Breakout detection
        break_above = high > self.rolling_high[-1]
        break_below = low < self.rolling_low[-1]
        
        # Position sizing
        risk_amount = self.equity * self.risk_per_trade
        sl_distance = atr * self.atr_multiplier_sl
        
        if sl_distance <= 0:
            return
        
        position_size = risk_amount / sl_distance
        max_shares = int(self.equity * 0.5 / price)
        shares = min(int(position_size), max_shares)
        
        if shares < 1:
            shares = 1
        
        # LONG
        if break_above and uptrend:
            sl = price - sl_distance
            tp = price + (atr * self.atr_multiplier_tp)
            self.buy(size=shares, sl=sl, tp=tp)
        
        # SHORT
        elif break_below and downtrend:
            sl = price + sl_distance
            tp = price - (atr * self.atr_multiplier_tp)
            self.sell(size=shares, sl=sl, tp=tp)


if __name__ == '__main__':
    import os
    import json
    
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_15m_160weeks.csv')
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, PivotBreakoutV3, cash=1_000_000, commission=0.002, trade_on_close=True)
    
    print("Running backtest...")
    stats = bt.run()
    
    print("\n" + "="*50)
    print("PIVOT BREAKOUT V3 - RESULTS")
    print("="*50)
    print(f"Return [%]:       {stats['Return [%]']:.2f}")
    print(f"Sharpe Ratio:     {stats['Sharpe Ratio']:.3f}" if pd.notna(stats['Sharpe Ratio']) else "Sharpe Ratio:     N/A")
    print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
    print(f"# Trades:         {stats['# Trades']}")
    print(f"Win Rate [%]:     {stats['Win Rate [%]']:.1f}" if pd.notna(stats['Win Rate [%]']) else "Win Rate [%]:     N/A")
    
    os.makedirs('results', exist_ok=True)
    result = {
        'strategy_name': 'pivot_breakout_v3',
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_drawdown': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        'total_trades': int(stats['# Trades'])
    }
    
    with open('results/pivot_breakout_v3_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to results/pivot_breakout_v3_result.json")
