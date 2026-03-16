"""
RSI Mean Reversion Strategy - Rewrite v5
==========================================
Final iteration: Simple, proven RSI mean reversion.
Based on the classic "buy oversold, sell overbought" concept.

Key features:
1. RSI < 30 for long, RSI > 70 for short
2. Wait for RSI to turn (confirmation)
3. Fixed 3:1 R:R with tight stops
4. Very conservative position sizing (0.5%)

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


class RSIMeanReversionV5(Strategy):
    """
    RSI Mean Reversion Strategy v5
    
    Entry Logic:
    - LONG: RSI < 30 and RSI is rising (oversold bounce)
    - SHORT: RSI > 70 and RSI is falling (overbought rejection)
    
    Risk Management:
    - Stop Loss: 1.5x ATR
    - Take Profit: 4.5x ATR (3:1 R:R)
    - Position Size: 0.5% equity risk
    """
    
    # Optimizable parameters
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    atr_period = 14
    atr_multiplier_sl = 1.5
    atr_multiplier_tp = 4.5
    risk_per_trade = 0.005  # 0.5% of equity
    
    def init(self):
        close = pd.Series(self.data.Close)
        
        # RSI
        rsi = ta.rsi(close, length=self.rsi_period)
        self.rsi = self.I(lambda: rsi.values)
        
        # ATR
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close,
                         self.atr_period)
    
    def next(self):
        if self.position:
            return
        if len(self.data) < 20:
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2] if len(self.rsi) > 1 else rsi
        
        if np.isnan(atr) or atr <= 0 or np.isnan(rsi) or np.isnan(rsi_prev):
            return
        
        # RSI conditions with confirmation
        oversold_bounce = rsi < self.rsi_oversold and rsi > rsi_prev  # RSI < 30 and rising
        overbought_drop = rsi > self.rsi_overbought and rsi < rsi_prev  # RSI > 70 and falling
        
        # Position sizing
        risk_amount = self.equity * self.risk_per_trade
        sl_distance = atr * self.atr_multiplier_sl
        
        if sl_distance <= 0:
            return
            
        position_size = risk_amount / sl_distance
        max_shares = int(self.equity * 0.5 / price)  # Cap at 50% of equity
        shares = min(int(position_size), max_shares)
        
        if shares < 1:
            shares = 1
        
        # LONG: Oversold bounce
        if oversold_bounce:
            sl = price - sl_distance
            tp = price + (atr * self.atr_multiplier_tp)
            self.buy(size=shares, sl=sl, tp=tp)
        
        # SHORT: Overbought drop
        elif overbought_drop:
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
    
    bt = Backtest(data, RSIMeanReversionV5, cash=1_000_000, commission=0.002, trade_on_close=True)
    
    print("Running backtest...")
    stats = bt.run()
    
    print("\n" + "="*50)
    print("RSI MEAN REVERSION V5 - RESULTS")
    print("="*50)
    print(f"Return [%]:       {stats['Return [%]']:.2f}")
    print(f"Sharpe Ratio:     {stats['Sharpe Ratio']:.3f}" if pd.notna(stats['Sharpe Ratio']) else "Sharpe Ratio:     N/A")
    print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
    print(f"# Trades:         {stats['# Trades']}")
    print(f"Win Rate [%]:     {stats['Win Rate [%]']:.1f}" if pd.notna(stats['Win Rate [%]']) else "Win Rate [%]:     N/A")
    
    os.makedirs('results', exist_ok=True)
    result = {
        'strategy_name': 'rsi_mean_reversion_v5',
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_drawdown': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        'total_trades': int(stats['# Trades'])
    }
    
    with open('results/rsi_mean_reversion_v5_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to results/rsi_mean_reversion_v5_result.json")
