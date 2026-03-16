"""
Momentum Reversal Strategy - Rewrite v3
========================================
Iteration 3: Added momentum filters to reduce whipsaws.
Changes from v2:
1. RSI momentum filter: RSI must be moving in favor direction
2. Volume confirmation: above average volume on entry
3. Wider stops: 2.5x ATR (was 1.5x)
4. Better R:R: 2:1 instead of 1.67:1

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


class MomentumReversalV3(Strategy):
    """
    Momentum Reversal Strategy v3
    
    Entry Logic (with momentum confirmation):
    - LONG: MACD bullish crossover + RSI rising + above avg volume
    - SHORT: MACD bearish crossover + RSI falling + above avg volume
    
    Risk Management:
    - Stop Loss: 2.5x ATR
    - Take Profit: 5x ATR (2:1 R:R)
    """
    
    # Optimizable parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    atr_period = 14
    atr_multiplier_sl = 2.5
    atr_multiplier_tp = 5.0
    volume_lookback = 20
    risk_per_trade = 0.015  # 1.5% of equity
    
    def init(self):
        close = pd.Series(self.data.Close)
        volume = pd.Series(self.data.Volume)
        
        # MACD
        macd_df = ta.macd(close=close, fast=self.macd_fast, 
                         slow=self.macd_slow, signal=self.macd_signal)
        self.macd_line = self.I(lambda: macd_df.iloc[:, 0].values)
        self.macd_signal_line = self.I(lambda: macd_df.iloc[:, 1].values)
        self.macd_hist = self.I(lambda: macd_df.iloc[:, 2].values)
        
        # RSI
        rsi = ta.rsi(close, length=self.rsi_period)
        self.rsi = self.I(lambda: rsi.values)
        
        # EMA for trend filter
        ema_50 = ta.ema(close, length=50)
        ema_200 = ta.ema(close, length=200)
        self.ema_50 = self.I(lambda: ema_50.values)
        self.ema_200 = self.I(lambda: ema_200.values)
        
        # Volume moving average
        vol_ma = volume.rolling(self.volume_lookback).mean()
        self.vol_ma = self.I(lambda: vol_ma.values)
        
        # ATR
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close,
                         self.atr_period)
    
    def next(self):
        if self.position:
            return
        if len(self.data) < 210:  # Need enough data for EMA200
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        volume = self.data.Volume[-1]
        
        if np.isnan(atr) or atr <= 0 or np.isnan(rsi):
            return
        
        # MACD crossover
        macd_bull_cross = crossover(self.macd_line, self.macd_signal_line)
        macd_bear_cross = crossover(self.macd_signal_line, self.macd_line)
        
        # RSI momentum: must be moving in the right direction
        rsi_prev = self.rsi[-2] if len(self.rsi) > 1 else rsi
        rsi_rising = rsi > rsi_prev
        rsi_falling = rsi < rsi_prev
        
        # Volume confirmation: above average
        vol_above_avg = volume > self.vol_ma[-1] if not np.isnan(self.vol_ma[-1]) else True
        
        # Trend: EMA alignment
        uptrend = self.ema_50[-1] > self.ema_200[-1]
        downtrend = self.ema_50[-1] < self.ema_200[-1]
        
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
        
        # LONG: MACD bullish crossover + RSI rising + volume + uptrend
        if macd_bull_cross and rsi_rising and vol_above_avg and uptrend:
            sl = price - sl_distance
            tp = price + (atr * self.atr_multiplier_tp)
            self.buy(size=shares, sl=sl, tp=tp)
        
        # SHORT: MACD bearish crossover + RSI falling + volume + downtrend
        elif macd_bear_cross and rsi_falling and vol_above_avg and downtrend:
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
    
    bt = Backtest(data, MomentumReversalV3, cash=1_000_000, commission=0.002, trade_on_close=True)
    
    print("Running backtest...")
    stats = bt.run()
    
    print("\n" + "="*50)
    print("MOMENTUM REVERSAL V3 - RESULTS")
    print("="*50)
    print(f"Return [%]:       {stats['Return [%]']:.2f}")
    print(f"Sharpe Ratio:     {stats['Sharpe Ratio']:.3f}" if pd.notna(stats['Sharpe Ratio']) else "Sharpe Ratio:     N/A")
    print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
    print(f"# Trades:         {stats['# Trades']}")
    print(f"Win Rate [%]:     {stats['Win Rate [%]']:.1f}" if pd.notna(stats['Win Rate [%]']) else "Win Rate [%]:     N/A")
    
    os.makedirs('results', exist_ok=True)
    result = {
        'strategy_name': 'momentum_reversal_v3',
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_drawdown': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        'total_trades': int(stats['# Trades'])
    }
    
    with open('results/momentum_reversal_v3_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to results/momentum_reversal_v3_result.json")
