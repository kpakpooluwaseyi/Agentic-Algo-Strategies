"""
Momentum Reversal Strategy - Rewrite v1
========================================
A simplified, robust momentum reversal strategy.
Improvements over original sr_macd_stochrsi_reversal:
1. Simplified entry: MACD crossover + RSI divergence (not 5+ conditions)
2. Proper recent S/R detection (20-bar lookback, not all-time extremes)
3. ATR-based dynamic stops (not fixed percentage)
4. Immediate entry (no 2-bar delay)
5. Position sizing based on risk

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
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(period).mean().values
    return atr


def find_recent_swing_high(high, lookback=20):
    """Find the most recent swing high within lookback period."""
    result = np.full(len(high), np.nan)
    for i in range(lookback, len(high)):
        window = high[i-lookback:i]
        result[i] = np.max(window)
    return result


def find_recent_swing_low(low, lookback=20):
    """Find the most recent swing low within lookback period."""
    result = np.full(len(low), np.nan)
    for i in range(lookback, len(low)):
        window = low[i-lookback:i]
        result[i] = np.min(window)
    return result


class MomentumReversalV1(Strategy):
    """
    Momentum Reversal Strategy v1
    
    Entry Logic:
    - LONG: MACD bullish crossover + RSI < 40 (oversold recovery)
    - SHORT: MACD bearish crossover + RSI > 60 (overbought rejection)
    
    Risk Management:
    - Stop Loss: 2x ATR
    - Take Profit: 3x ATR (1.5:1 R:R)
    - Position Size: 1% equity risk per trade
    """
    
    # Optimizable parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    atr_period = 14
    atr_multiplier_sl = 2.0
    atr_multiplier_tp = 3.0
    rsi_oversold = 40
    rsi_overbought = 60
    risk_per_trade = 0.01  # 1% of equity
    
    def init(self):
        close = pd.Series(self.data.Close)
        
        # MACD
        macd_df = ta.macd(close=close, fast=self.macd_fast, 
                         slow=self.macd_slow, signal=self.macd_signal)
        self.macd_line = self.I(lambda: macd_df.iloc[:, 0].values)
        self.macd_signal_line = self.I(lambda: macd_df.iloc[:, 1].values)
        self.macd_hist = self.I(lambda: macd_df.iloc[:, 2].values)
        
        # RSI
        rsi = ta.rsi(close, length=self.rsi_period)
        self.rsi = self.I(lambda: rsi.values)
        
        # ATR for position sizing and stops
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close,
                         self.atr_period)
        
        # Recent swing levels for reference
        self.swing_high = self.I(find_recent_swing_high, self.data.High, 20)
        self.swing_low = self.I(find_recent_swing_low, self.data.Low, 20)
    
    def next(self):
        # Skip if already in position or insufficient data
        if self.position:
            return
        if len(self.data) < self.macd_slow + self.macd_signal:
            return
        
        # Get current values
        price = self.data.Close[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        
        # Skip if ATR is invalid
        if np.isnan(atr) or atr <= 0:
            return
        
        # MACD crossover detection
        macd_bull_cross = crossover(self.macd_line, self.macd_signal_line)
        macd_bear_cross = crossover(self.macd_signal_line, self.macd_line)
        
        # Calculate position size based on risk
        risk_amount = self.equity * self.risk_per_trade
        sl_distance = atr * self.atr_multiplier_sl
        
        # Avoid division by zero
        if sl_distance <= 0:
            return
            
        # Position size: risk_amount / stop_distance
        position_size = risk_amount / sl_distance
        
        # Cap position size to affordable amount
        max_shares = int(self.equity * 0.95 / price)  # 95% of equity max
        shares = min(int(position_size), max_shares)
        
        if shares < 1:
            shares = 1  # Minimum 1 share
        
        # LONG: MACD bullish crossover + RSI recovering from oversold
        if macd_bull_cross and rsi < self.rsi_oversold:
            sl = price - sl_distance
            tp = price + (atr * self.atr_multiplier_tp)
            self.buy(size=shares, sl=sl, tp=tp)
        
        # SHORT: MACD bearish crossover + RSI rejecting from overbought
        elif macd_bear_cross and rsi > self.rsi_overbought:
            sl = price + sl_distance
            tp = price - (atr * self.atr_multiplier_tp)
            self.sell(size=shares, sl=sl, tp=tp)


# Standalone execution
if __name__ == '__main__':
    import os
    import json
    
    # Try standardized mode first
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_15m_160weeks.csv')
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    # Use FractionalBacktest if available
    try:
        from backtesting.lib import FractionalBacktest
        bt = FractionalBacktest(data, MomentumReversalV1, cash=1_000_000, commission=0.002)
    except ImportError:
        bt = Backtest(data, MomentumReversalV1, cash=1_000_000, commission=0.002)
    
    print("Running backtest...")
    stats = bt.run()
    
    # Print key metrics
    print("\n" + "="*50)
    print("MOMENTUM REVERSAL V1 - RESULTS")
    print("="*50)
    print(f"Return [%]:       {stats['Return [%]']:.2f}")
    print(f"Sharpe Ratio:     {stats['Sharpe Ratio']:.3f}" if pd.notna(stats['Sharpe Ratio']) else "Sharpe Ratio:     N/A")
    print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
    print(f"# Trades:         {stats['# Trades']}")
    print(f"Win Rate [%]:     {stats['Win Rate [%]']:.1f}" if pd.notna(stats['Win Rate [%]']) else "Win Rate [%]:     N/A")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    result = {
        'strategy_name': 'momentum_reversal_v1',
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_drawdown': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        'total_trades': int(stats['# Trades'])
    }
    
    with open('results/momentum_reversal_v1_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to results/momentum_reversal_v1_result.json")
