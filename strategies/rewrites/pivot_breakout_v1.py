"""
Pivot Breakout Strategy - Rewrite v1
=====================================
Removed moon phase dependency from bitcoin_monthly_moon_pivot.
Keep the pivot detection concept, add trend confirmation.

Key changes:
1. No moon phases - just pure pivot detection
2. Added EMA trend filter
3. Simplified to look for swing high/low breakouts
4. Conservative 3:1 R:R with ATR stops

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


class PivotBreakoutV1(Strategy):
    """
    Pivot Breakout Strategy v1
    
    Detects swing pivots and trades breakouts with trend confirmation.
    
    Entry Logic:
    - LONG: Break above pivot high + price > EMA50
    - SHORT: Break below pivot low + price < EMA50
    
    Risk Management:
    - Stop Loss: Pivot level (the level we broke)
    - Take Profit: 3x risk
    """
    
    # Optimizable parameters
    pivot_lookback = 5  # Bars to look for pivot
    atr_period = 14
    risk_reward = 3.0
    risk_per_trade = 0.01
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # EMA for trend filter
        ema_50 = ta.ema(close, length=50)
        self.ema_50 = self.I(lambda: ema_50.values)
        
        # Rolling highs/lows for pivot detection
        self.pivot_high = self.I(lambda: high.rolling(self.pivot_lookback * 2 + 1, center=True).max().values)
        self.pivot_low = self.I(lambda: low.rolling(self.pivot_lookback * 2 + 1, center=True).min().values)
        
        # ATR
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close,
                         self.atr_period)
        
        # Track last pivot levels
        self.last_pivot_high = None
        self.last_pivot_low = None
    
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
        
        # Update pivot levels (looking back a few bars for confirmed pivots)
        lookback_offset = self.pivot_lookback + 1
        if len(self.data) > lookback_offset:
            # Check if bar at lookback was a pivot high
            if self.data.High[-lookback_offset] == self.pivot_high[-lookback_offset]:
                self.last_pivot_high = self.data.High[-lookback_offset]
            # Check if bar at lookback was a pivot low
            if self.data.Low[-lookback_offset] == self.pivot_low[-lookback_offset]:
                self.last_pivot_low = self.data.Low[-lookback_offset]
        
        if self.last_pivot_high is None or self.last_pivot_low is None:
            return
        
        # Trend filter
        uptrend = price > self.ema_50[-1]
        downtrend = price < self.ema_50[-1]
        
        # Breakout detection
        break_above_pivot_high = high > self.last_pivot_high and self.data.High[-2] <= self.last_pivot_high
        break_below_pivot_low = low < self.last_pivot_low and self.data.Low[-2] >= self.last_pivot_low
        
        # Position sizing
        risk_amount = self.equity * self.risk_per_trade
        
        # LONG: Break above pivot high in uptrend
        if break_above_pivot_high and uptrend:
            sl = self.last_pivot_low  # Stop at recent pivot low
            risk = price - sl
            if risk > 0:
                tp = price + risk * self.risk_reward
                position_size = risk_amount / risk
                max_shares = int(self.equity * 0.5 / price)
                shares = min(int(position_size), max_shares)
                if shares >= 1:
                    self.buy(size=shares, sl=sl, tp=tp)
                    self.last_pivot_high = None  # Reset after trade
        
        # SHORT: Break below pivot low in downtrend
        elif break_below_pivot_low and downtrend:
            sl = self.last_pivot_high  # Stop at recent pivot high
            risk = sl - price
            if risk > 0:
                tp = price - risk * self.risk_reward
                position_size = risk_amount / risk
                max_shares = int(self.equity * 0.5 / price)
                shares = min(int(position_size), max_shares)
                if shares >= 1:
                    self.sell(size=shares, sl=sl, tp=tp)
                    self.last_pivot_low = None  # Reset after trade


if __name__ == '__main__':
    import os
    import json
    
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_15m_160weeks.csv')
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, PivotBreakoutV1, cash=1_000_000, commission=0.002, trade_on_close=True)
    
    print("Running backtest...")
    stats = bt.run()
    
    print("\n" + "="*50)
    print("PIVOT BREAKOUT V1 - RESULTS")
    print("="*50)
    print(f"Return [%]:       {stats['Return [%]']:.2f}")
    print(f"Sharpe Ratio:     {stats['Sharpe Ratio']:.3f}" if pd.notna(stats['Sharpe Ratio']) else "Sharpe Ratio:     N/A")
    print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
    print(f"# Trades:         {stats['# Trades']}")
    print(f"Win Rate [%]:     {stats['Win Rate [%]']:.1f}" if pd.notna(stats['Win Rate [%]']) else "Win Rate [%]:     N/A")
    
    os.makedirs('results', exist_ok=True)
    result = {
        'strategy_name': 'pivot_breakout_v1',
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_drawdown': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        'total_trades': int(stats['# Trades'])
    }
    
    with open('results/pivot_breakout_v1_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to results/pivot_breakout_v1_result.json")
