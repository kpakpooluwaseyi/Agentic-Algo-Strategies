"""
Measured Move Breakout Strategy - Rewrite from 50_50_mow_internal_scalp
========================================================================
Original Issue: Zero trades due to overly complex M-pattern detection

New Logic:
1. Detect swing highs/lows using simple rolling windows
2. Enter on 50% retracement of prior move
3. Target measured move projection
4. Use 4H timeframe (proven to work from earlier tests)

Author: Antigravity (Claude Opus)
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import os
import json


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(period).mean().values
    return atr


class MeasuredMoveBreakout(Strategy):
    """
    Measured Move Breakout Strategy
    
    Detects swing high/low and enters on 50% retracement.
    Targets 100% measured move projection.
    
    Entry:
    - LONG: Price retraces 50% of prior upswing + turns up
    - SHORT: Price retraces 50% of prior downswing + turns down
    
    Exit:
    - TP: 100% measured move (full leg projection)
    - SL: Beyond the swing extreme
    """
    
    swing_period = 10  # Bars to detect swing
    retracement_zone = 0.05  # 5% tolerance around 50% level
    risk_per_trade = 0.02
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Rolling highs/lows for swing detection
        self.swing_high = self.I(lambda: high.rolling(self.swing_period * 2 + 1, center=True).max().values)
        self.swing_low = self.I(lambda: low.rolling(self.swing_period * 2 + 1, center=True).min().values)
        
        # EMA for trend filter
        ema_50 = ta.ema(close, length=50)
        self.ema_50 = self.I(lambda: ema_50.values)
        
        # ATR for volatility
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close, 14)
        
        # Track last confirmed swings
        self.last_swing_high = None
        self.last_swing_high_idx = 0
        self.last_swing_low = None
        self.last_swing_low_idx = 0
    
    def next(self):
        if self.position:
            return
        if len(self.data) < 30:
            return
        
        price = self.data.Close[-1]
        idx = len(self.data) - 1
        
        # Update swing points (looking back a bit to confirm)
        lookback = self.swing_period + 1
        if len(self.data) > lookback:
            if self.data.High[-lookback] == self.swing_high[-lookback]:
                self.last_swing_high = self.data.High[-lookback]
                self.last_swing_high_idx = idx - lookback
            if self.data.Low[-lookback] == self.swing_low[-lookback]:
                self.last_swing_low = self.data.Low[-lookback]
                self.last_swing_low_idx = idx - lookback
        
        if self.last_swing_high is None or self.last_swing_low is None:
            return
        
        atr = self.atr[-1]
        if np.isnan(atr) or atr <= 0:
            return
        
        # Determine the current swing structure
        # If last swing high is more recent than swing low -> we're in a pullback of uptrend
        in_uptrend_pullback = self.last_swing_high_idx > self.last_swing_low_idx
        # If last swing low is more recent than swing high -> we're in a pullback of downtrend
        in_downtrend_pullback = self.last_swing_low_idx > self.last_swing_high_idx
        
        # Calculate 50% level
        if in_uptrend_pullback:
            # Pullback from high towards low
            move_size = self.last_swing_high - self.last_swing_low
            fib_50 = self.last_swing_high - (move_size * 0.5)
            fib_tolerance = move_size * self.retracement_zone
            
            # Check if price is in 50% zone
            in_zone = abs(price - fib_50) <= fib_tolerance
            
            # Entry: price in zone + turning up (current bar higher than prev)
            turning_up = self.data.Close[-1] > self.data.Open[-1]
            uptrend = price > self.ema_50[-1] if not np.isnan(self.ema_50[-1]) else True
            
            if in_zone and turning_up and uptrend:
                # SL below swing low, TP at measured move above swing high
                sl = self.last_swing_low - (atr * 0.5)
                tp = self.last_swing_high + move_size  # 100% extension
                
                risk = price - sl
                if risk > 0:
                    shares = int((self.equity * self.risk_per_trade) / risk)
                    shares = max(1, min(shares, int(self.equity * 0.5 / price)))
                    self.buy(size=shares, sl=sl, tp=tp)
        
        elif in_downtrend_pullback:
            # Pullback from low towards high
            move_size = self.last_swing_high - self.last_swing_low
            fib_50 = self.last_swing_low + (move_size * 0.5)
            fib_tolerance = move_size * self.retracement_zone
            
            # Check if price is in 50% zone
            in_zone = abs(price - fib_50) <= fib_tolerance
            
            # Entry: price in zone + turning down (current bar lower than prev)
            turning_down = self.data.Close[-1] < self.data.Open[-1]
            downtrend = price < self.ema_50[-1] if not np.isnan(self.ema_50[-1]) else True
            
            if in_zone and turning_down and downtrend:
                # SL above swing high, TP at measured move below swing low
                sl = self.last_swing_high + (atr * 0.5)
                tp = self.last_swing_low - move_size  # 100% extension
                
                risk = sl - price
                if risk > 0:
                    shares = int((self.equity * self.risk_per_trade) / risk)
                    shares = max(1, min(shares, int(self.equity * 0.5 / price)))
                    self.sell(size=shares, sl=sl, tp=tp)


if __name__ == '__main__':
    # Test on 4H BTC (proven timeframe)
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_4h_200weeks.csv')
    
    print("="*60)
    print("MEASURED MOVE BREAKOUT - STRATEGY 3 TEST")
    print("="*60)
    print(f"Dataset: {data_path}")
    
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, MeasuredMoveBreakout, cash=1_000_000, 
                 commission=0.002, trade_on_close=True)
    stats = bt.run()
    
    print("\n" + "-"*60)
    print("RESULTS:")
    print("-"*60)
    print(f"Return [%]:       {stats['Return [%]']:.2f}")
    print(f"Sharpe Ratio:     {stats['Sharpe Ratio']:.3f}" if pd.notna(stats['Sharpe Ratio']) else "Sharpe Ratio:     N/A")
    print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
    print(f"# Trades:         {stats['# Trades']}")
    print(f"Win Rate [%]:     {stats['Win Rate [%]']:.1f}" if pd.notna(stats['Win Rate [%]']) else "Win Rate [%]:     N/A")
    
    os.makedirs('results', exist_ok=True)
    result = {
        'strategy_name': 'measured_move_breakout',
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_drawdown': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        'total_trades': int(stats['# Trades'])
    }
    
    with open('results/measured_move_breakout_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    profitable = "✅ PROFITABLE" if result['return'] > 0 else "❌ NEEDS ITERATION"
    print(f"\n{profitable}")
