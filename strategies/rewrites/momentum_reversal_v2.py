"""
Momentum Reversal Strategy - Rewrite v2
========================================
Iteration 2: Relaxed entry conditions for more trades.
Changes from v1:
1. RSI threshold relaxed: < 50 for long, > 50 for short
2. Added MACD histogram momentum confirmation
3. Reduced ATR multipliers for tighter stops
4. Simplified: just MACD crossover with trend filter

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
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


class MomentumReversalV2(Strategy):
    """
    Momentum Reversal Strategy v2
    
    Entry Logic (Simplified):
    - LONG: MACD bullish crossover + histogram rising
    - SHORT: MACD bearish crossover + histogram falling
    
    Risk Management:
    - Stop Loss: 1.5x ATR
    - Take Profit: 2.5x ATR
    """
    
    # Optimizable parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    atr_multiplier_sl = 1.5
    atr_multiplier_tp = 2.5
    risk_per_trade = 0.02  # 2% of equity
    
    def init(self):
        close = pd.Series(self.data.Close)
        
        # MACD
        macd_df = ta.macd(close=close, fast=self.macd_fast, 
                         slow=self.macd_slow, signal=self.macd_signal)
        self.macd_line = self.I(lambda: macd_df.iloc[:, 0].values)
        self.macd_signal_line = self.I(lambda: macd_df.iloc[:, 1].values)
        self.macd_hist = self.I(lambda: macd_df.iloc[:, 2].values)
        
        # EMA for trend filter
        ema_50 = ta.ema(close, length=50)
        self.ema_50 = self.I(lambda: ema_50.values)
        
        # ATR
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close,
                         self.atr_period)
    
    def next(self):
        if self.position:
            return
        if len(self.data) < 60:  # Need enough data for EMA
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        # MACD crossover
        macd_bull_cross = crossover(self.macd_line, self.macd_signal_line)
        macd_bear_cross = crossover(self.macd_signal_line, self.macd_line)
        
        # Histogram momentum
        hist_rising = self.macd_hist[-1] > self.macd_hist[-2]
        hist_falling = self.macd_hist[-1] < self.macd_hist[-2]
        
        # Trend filter: price vs EMA
        above_ema = price > self.ema_50[-1]
        below_ema = price < self.ema_50[-1]
        
        # Position sizing
        risk_amount = self.equity * self.risk_per_trade
        sl_distance = atr * self.atr_multiplier_sl
        
        if sl_distance <= 0:
            return
            
        position_size = risk_amount / sl_distance
        max_shares = int(self.equity * 0.95 / price)
        shares = min(int(position_size), max_shares)
        
        if shares < 1:
            shares = 1
        
        # LONG: MACD bullish crossover + histogram rising + above EMA
        if macd_bull_cross and hist_rising and above_ema:
            sl = price - sl_distance
            tp = price + (atr * self.atr_multiplier_tp)
            self.buy(size=shares, sl=sl, tp=tp)
        
        # SHORT: MACD bearish crossover + histogram falling + below EMA
        elif macd_bear_cross and hist_falling and below_ema:
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
    
    try:
        from backtesting.lib import FractionalBacktest
        bt = Backtest(data, MomentumReversalV2, cash=1_000_000, commission=0.002, trade_on_close=True)
    except ImportError:
        bt = Backtest(data, MomentumReversalV2, cash=1_000_000, commission=0.002, trade_on_close=True)
    
    print("Running backtest...")
    stats = bt.run()
    
    print("\n" + "="*50)
    print("MOMENTUM REVERSAL V2 - RESULTS")
    print("="*50)
    print(f"Return [%]:       {stats['Return [%]']:.2f}")
    print(f"Sharpe Ratio:     {stats['Sharpe Ratio']:.3f}" if pd.notna(stats['Sharpe Ratio']) else "Sharpe Ratio:     N/A")
    print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
    print(f"# Trades:         {stats['# Trades']}")
    print(f"Win Rate [%]:     {stats['Win Rate [%]']:.1f}" if pd.notna(stats['Win Rate [%]']) else "Win Rate [%]:     N/A")
    
    os.makedirs('results', exist_ok=True)
    result = {
        'strategy_name': 'momentum_reversal_v2',
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_drawdown': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        'total_trades': int(stats['# Trades'])
    }
    
    with open('results/momentum_reversal_v2_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to results/momentum_reversal_v2_result.json")
