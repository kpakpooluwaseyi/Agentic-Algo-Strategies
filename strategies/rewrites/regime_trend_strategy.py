"""
Regime Trend Strategy
======================
Long/Short strategy with volatility regime filter.
Trades both directions based on trend strength and volatility conditions.

Regime Filter:
- ATR Percentile: Only trades when volatility is "normal" (25th-75th percentile)
- Avoids extreme volatility (whipsaws) and dead markets (low opportunity)

Entry Logic:
- LONG: EMA crossover + price > EMA200 + RSI > 50 + normal volatility
- SHORT: EMA crossover (bearish) + price < EMA200 + RSI < 50 + normal volatility

Risk Management:
- Stop Loss: 1.5x ATR
- Take Profit: 3x ATR (2:1 R:R)
- Position Size: 1% equity risk

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


def calculate_atr_percentile(atr, lookback=100):
    """Calculate where current ATR sits in recent history (0-100)."""
    result = np.full(len(atr), np.nan)
    for i in range(lookback, len(atr)):
        window = atr[i-lookback:i+1]
        if len(window) > 0 and not np.all(np.isnan(window)):
            # Percentile rank of current ATR
            current = atr[i]
            count_below = np.sum(window < current)
            result[i] = (count_below / len(window)) * 100
    return result


class RegimeTrendStrategy(Strategy):
    """
    Regime-filtered trend strategy.
    Trades long and short with volatility regime filtering.
    """
    
    # EMA parameters
    ema_fast = 20
    ema_slow = 50
    ema_trend = 200
    
    # Regime filter parameters
    atr_period = 14
    atr_lookback = 100
    atr_low_percentile = 25   # Min volatility percentile
    atr_high_percentile = 75  # Max volatility percentile
    
    # Risk parameters
    rsi_period = 14
    atr_multiplier_sl = 1.5
    atr_multiplier_tp = 3.0
    risk_per_trade = 0.01
    
    def init(self):
        close = pd.Series(self.data.Close)
        
        # EMAs
        ema_fast = ta.ema(close, length=self.ema_fast)
        ema_slow = ta.ema(close, length=self.ema_slow)
        ema_trend = ta.ema(close, length=self.ema_trend)
        
        self.ema_fast = self.I(lambda: ema_fast.values)
        self.ema_slow = self.I(lambda: ema_slow.values)
        self.ema_trend = self.I(lambda: ema_trend.values)
        
        # RSI
        rsi = ta.rsi(close, length=self.rsi_period)
        self.rsi = self.I(lambda: rsi.values)
        
        # ATR and regime
        self.atr = self.I(calculate_atr, 
                         self.data.High, self.data.Low, self.data.Close,
                         self.atr_period)
        
        # ATR percentile for regime detection
        atr_values = calculate_atr(
            np.array(self.data.High), 
            np.array(self.data.Low), 
            np.array(self.data.Close), 
            self.atr_period
        )
        self.atr_percentile = self.I(calculate_atr_percentile, atr_values, self.atr_lookback)
    
    def next(self):
        if self.position:
            return
        if len(self.data) < 220:  # Need enough data for EMA200 + lookback
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        atr_pct = self.atr_percentile[-1]
        
        # Validate data
        if np.isnan(atr) or atr <= 0 or np.isnan(rsi) or np.isnan(atr_pct):
            return
        
        # REGIME FILTER: Only trade in normal volatility
        regime_ok = self.atr_low_percentile <= atr_pct <= self.atr_high_percentile
        if not regime_ok:
            return
        
        # Trend direction
        uptrend = price > self.ema_trend[-1]
        downtrend = price < self.ema_trend[-1]
        
        # EMA crossovers
        ema_bull_cross = crossover(self.ema_fast, self.ema_slow)
        ema_bear_cross = crossover(self.ema_slow, self.ema_fast)
        
        # Momentum confirmation
        bullish_momentum = rsi > 50
        bearish_momentum = rsi < 50
        
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
        
        # LONG: Uptrend + EMA bull cross + bullish momentum + normal volatility
        if uptrend and ema_bull_cross and bullish_momentum:
            sl = price - sl_distance
            tp = price + (atr * self.atr_multiplier_tp)
            self.buy(size=shares, sl=sl, tp=tp)
        
        # SHORT: Downtrend + EMA bear cross + bearish momentum + normal volatility
        elif downtrend and ema_bear_cross and bearish_momentum:
            sl = price + sl_distance
            tp = price - (atr * self.atr_multiplier_tp)
            self.sell(size=shares, sl=sl, tp=tp)


if __name__ == '__main__':
    import os
    import json
    
    # Use 2025 BTC data
    data_path = os.environ.get('BACKTEST_DATA_PATH', 'data/crypto/BTCUSDT_P_15m_2025.csv')
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    data.columns = [c.strip().capitalize() for c in data.columns]
    
    bt = Backtest(data, RegimeTrendStrategy, cash=1_000_000, commission=0.002, trade_on_close=True)
    
    print("Running backtest...")
    stats = bt.run()
    
    print("\n" + "="*60)
    print("REGIME TREND STRATEGY - 2025 BTC 15m")
    print("="*60)
    print(f"Return [%]:       {stats['Return [%]']:.2f}")
    print(f"Sharpe Ratio:     {stats['Sharpe Ratio']:.3f}" if pd.notna(stats['Sharpe Ratio']) else "Sharpe Ratio:     N/A")
    print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
    print(f"# Trades:         {stats['# Trades']}")
    print(f"Win Rate [%]:     {stats['Win Rate [%]']:.1f}" if pd.notna(stats['Win Rate [%]']) else "Win Rate [%]:     N/A")
    
    os.makedirs('results', exist_ok=True)
    result = {
        'strategy_name': 'regime_trend_strategy',
        'dataset': data_path,
        'return': float(stats['Return [%]']) if pd.notna(stats['Return [%]']) else 0,
        'sharpe': float(stats['Sharpe Ratio']) if pd.notna(stats['Sharpe Ratio']) else 0,
        'max_drawdown': float(stats['Max. Drawdown [%]']) if pd.notna(stats['Max. Drawdown [%]']) else 0,
        'win_rate': float(stats['Win Rate [%]']) if pd.notna(stats['Win Rate [%]']) else 0,
        'total_trades': int(stats['# Trades'])
    }
    
    with open('results/regime_trend_strategy_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to results/regime_trend_strategy_result.json")
