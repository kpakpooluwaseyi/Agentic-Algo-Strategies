"""
Donchian Channel Trend Strategy
================================
Based on the Turtle Trading system.
Long/Short with strong trend regime filtering.

Regime Filter:
- Only trade when EMA200 slope is positive (long) or negative (short)
- Skip when ADX < 20 (no trend)

Entry Logic:
- LONG: Break above 20-bar high + uptrend confirmed
- SHORT: Break below 20-bar low + downtrend confirmed

Exit:
- LONG EXIT: Break below 10-bar low
- SHORT EXIT: Break above 10-bar high

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


class DonchianTrendStrategy(Strategy):
    """
    Donchian Channel breakout with trend filter.
    """
    
    # Channel parameters
    entry_period = 20   # 20-bar breakout for entry
    exit_period = 10    # 10-bar breakout for exit
    
    # Regime filter
    ema_period = 200
    ema_slope_lookback = 10  # Bars to measure EMA slope
    adx_period = 14
    adx_threshold = 20
    
    # Risk
    atr_period = 14
    risk_per_trade = 0.01
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Donchian Channels
        self.entry_high = self.I(lambda: high.shift(1).rolling(self.entry_period).max().values)
        self.entry_low = self.I(lambda: low.shift(1).rolling(self.entry_period).min().values)
        self.exit_high = self.I(lambda: high.shift(1).rolling(self.exit_period).max().values)
        self.exit_low = self.I(lambda: low.shift(1).rolling(self.exit_period).min().values)
        
        # Trend filter - EMA and its slope
        ema_200 = ta.ema(close, length=self.ema_period)
        self.ema_200 = self.I(lambda: ema_200.values)
        
        # ADX for trend strength
        adx = ta.adx(high, low, close, length=self.adx_period)
        self.adx = self.I(lambda: adx.iloc[:, 0].values)  # ADX column
        
        # ATR
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close,
                         self.atr_period)
    
    def next(self):
        if len(self.data) < 220:
            return
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        adx = self.adx[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        # Calculate EMA slope (trend direction strength)
        ema_now = self.ema_200[-1]
        ema_before = self.ema_200[-self.ema_slope_lookback] if len(self.ema_200) > self.ema_slope_lookback else ema_now
        ema_slope_up = ema_now > ema_before
        ema_slope_down = ema_now < ema_before
        
        # Trend confirmation
        strong_uptrend = ema_slope_up and (not np.isnan(adx) and adx > self.adx_threshold)
        strong_downtrend = ema_slope_down and (not np.isnan(adx) and adx > self.adx_threshold)
        
        # Manage existing position
        if self.position:
            if self.position.is_long:
                # Exit long: break below exit_low
                if low < self.exit_low[-1]:
                    self.position.close()
            elif self.position.is_short:
                # Exit short: break above exit_high
                if high > self.exit_high[-1]:
                    self.position.close()
            return
        
        # Entry signals
        break_above = high > self.entry_high[-1]
        break_below = low < self.entry_low[-1]
        
        # Position sizing: 1% risk based on ATR
        risk_amount = self.equity * self.risk_per_trade
        sl_distance = atr * 2  # 2x ATR stop
        
        if sl_distance <= 0:
            return
        
        position_size = risk_amount / sl_distance
        max_shares = int(self.equity * 0.5 / price)
        shares = min(int(position_size), max_shares)
        
        if shares < 1:
            shares = 1
        
        # LONG: Break above 20-bar high + strong uptrend
        if break_above and strong_uptrend:
            sl = price - sl_distance
            self.buy(size=shares, sl=sl)
        
        # SHORT: Break below 20-bar low + strong downtrend
        elif break_below and strong_downtrend:
            sl = price + sl_distance
            self.sell(size=shares, sl=sl)


if __name__ == '__main__':
    import os
    import json
    
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTC-USDT_15m_160weeks.csv')
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, DonchianTrendStrategy, cash=1_000_000, commission=0.002, trade_on_close=True)
    
    print("Running backtest...")
    stats = bt.run()
    
    print("\n" + "="*60)
    print("DONCHIAN TREND STRATEGY - BTC 15m (2022-2025)")
    print("="*60)
    print(f"Return [%]:       {stats['Return [%]']:.2f}")
    print(f"Sharpe Ratio:     {stats['Sharpe Ratio']:.3f}" if pd.notna(stats['Sharpe Ratio']) else "Sharpe Ratio:     N/A")
    print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
    print(f"# Trades:         {stats['# Trades']}")
    print(f"Win Rate [%]:     {stats['Win Rate [%]']:.1f}" if pd.notna(stats['Win Rate [%]']) else "Win Rate [%]:     N/A")
    
    os.makedirs('results', exist_ok=True)
    result = {
        'strategy_name': 'donchian_trend_strategy',
        'dataset': data_path,
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_drawdown': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        'total_trades': int(stats['# Trades'])
    }
    
    with open('results/donchian_trend_strategy_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to results/donchian_trend_strategy_result.json")
