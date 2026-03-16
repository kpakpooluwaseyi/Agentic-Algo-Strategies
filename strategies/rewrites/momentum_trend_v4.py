"""
Momentum Trend Strategy - Rewrite v4
=====================================
Iteration 4: Switched from reversal to TREND-FOLLOWING.
The 15m BTC market is trending, not mean-reverting.

Key changes:
1. Trade WITH the trend, not against it
2. MACD histogram divergence for early entry
3. Price breakout confirmation (new 20-bar high/low)
4. Trailing stop instead of fixed TP

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, TrailingStrategy
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


class MomentumTrendV4(Strategy):
    """
    Momentum Trend Strategy v4
    
    Entry Logic (TREND-FOLLOWING):
    - LONG: Price makes new 20-bar high + MACD > signal + EMA50 > EMA200
    - SHORT: Price makes new 20-bar low + MACD < signal + EMA50 < EMA200
    
    Risk Management:
    - Stop Loss: 2x ATR trailing stop
    - Exit: Trailing stop or trend reversal
    """
    
    # Optimizable parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    breakout_period = 20
    atr_period = 14
    atr_multiplier = 2.0
    risk_per_trade = 0.02
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # MACD
        macd_df = ta.macd(close=close, fast=self.macd_fast, 
                         slow=self.macd_slow, signal=self.macd_signal)
        self.macd_line = self.I(lambda: macd_df.iloc[:, 0].values)
        self.macd_signal_line = self.I(lambda: macd_df.iloc[:, 1].values)
        
        # EMAs for trend direction
        ema_50 = ta.ema(close, length=50)
        ema_200 = ta.ema(close, length=200)
        self.ema_50 = self.I(lambda: ema_50.values)
        self.ema_200 = self.I(lambda: ema_200.values)
        
        # Rolling highs/lows for breakout detection
        rolling_high = high.rolling(self.breakout_period).max()
        rolling_low = low.rolling(self.breakout_period).min()
        self.rolling_high = self.I(lambda: rolling_high.values)
        self.rolling_low = self.I(lambda: rolling_low.values)
        
        # ATR
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close,
                         self.atr_period)
        
        # Track trailing stop
        self.trailing_stop = None
        self.position_direction = None
    
    def next(self):
        if len(self.data) < 210:
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        # Manage existing position with trailing stop
        if self.position:
            if self.position.is_long:
                # Update trailing stop for long
                new_stop = price - (atr * self.atr_multiplier)
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                
                # Exit if price breaks below trailing stop or trend reverses
                if price < self.trailing_stop or self.ema_50[-1] < self.ema_200[-1]:
                    self.position.close()
                    self.trailing_stop = None
                    
            elif self.position.is_short:
                # Update trailing stop for short
                new_stop = price + (atr * self.atr_multiplier)
                if self.trailing_stop is None or new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                
                # Exit if price breaks above trailing stop or trend reverses
                if price > self.trailing_stop or self.ema_50[-1] > self.ema_200[-1]:
                    self.position.close()
                    self.trailing_stop = None
            return
        
        # Entry conditions
        uptrend = self.ema_50[-1] > self.ema_200[-1]
        downtrend = self.ema_50[-1] < self.ema_200[-1]
        
        macd_bullish = self.macd_line[-1] > self.macd_signal_line[-1]
        macd_bearish = self.macd_line[-1] < self.macd_signal_line[-1]
        
        # Breakout detection: price at or above rolling high/below rolling low
        at_high = self.data.High[-1] >= self.rolling_high[-2]  # Compare to previous bar's rolling high
        at_low = self.data.Low[-1] <= self.rolling_low[-2]
        
        # Position sizing
        risk_amount = self.equity * self.risk_per_trade
        sl_distance = atr * self.atr_multiplier
        
        if sl_distance <= 0:
            return
            
        position_size = risk_amount / sl_distance
        max_shares = int(self.equity * 0.8 / price)
        shares = min(int(position_size), max_shares)
        
        if shares < 1:
            shares = 1
        
        # LONG: Uptrend + MACD bullish + breakout
        if uptrend and macd_bullish and at_high:
            sl = price - sl_distance
            self.buy(size=shares, sl=sl)
            self.trailing_stop = sl
        
        # SHORT: Downtrend + MACD bearish + breakout
        elif downtrend and macd_bearish and at_low:
            sl = price + sl_distance
            self.sell(size=shares, sl=sl)
            self.trailing_stop = sl


if __name__ == '__main__':
    import os
    import json
    
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_15m_160weeks.csv')
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, MomentumTrendV4, cash=1_000_000, commission=0.002, trade_on_close=True)
    
    print("Running backtest...")
    stats = bt.run()
    
    print("\n" + "="*50)
    print("MOMENTUM TREND V4 - RESULTS")
    print("="*50)
    print(f"Return [%]:       {stats['Return [%]']:.2f}")
    print(f"Sharpe Ratio:     {stats['Sharpe Ratio']:.3f}" if pd.notna(stats['Sharpe Ratio']) else "Sharpe Ratio:     N/A")
    print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
    print(f"# Trades:         {stats['# Trades']}")
    print(f"Win Rate [%]:     {stats['Win Rate [%]']:.1f}" if pd.notna(stats['Win Rate [%]']) else "Win Rate [%]:     N/A")
    
    os.makedirs('results', exist_ok=True)
    result = {
        'strategy_name': 'momentum_trend_v4',
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_drawdown': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        'total_trades': int(stats['# Trades'])
    }
    
    with open('results/momentum_trend_v4_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to results/momentum_trend_v4_result.json")
