# Strategy Development Guidelines

**Version:** 1.0  
**Last Updated:** 2026-01-04

---

## Purpose

This document defines **mandatory requirements** for all trading strategy implementations to prevent overfitting and ensure robust, generalizable strategies that pass Walk-Forward Analysis (WFA) validation.

---

## Core Principles

1. **Adaptive, Not Static**: All parameters must adapt to market conditions
2. **Multi-Timeframe Context**: Always consider higher timeframe structure
3. **Volatility-Aware**: Use ATR for all risk management decisions
4. **Regime-Aware**: Adjust behavior based on market regime
5. **Volume-Confirmed**: Require volume confirmation for entries

---

## Required Features

### 1. ATR-Based Risk Management ✅

**Stop Loss:**
```python
atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
stop_loss = entry_price - (2 * atr[-1])  # For longs
stop_loss = entry_price + (2 * atr[-1])  # For shorts
```

**Take Profit:**
```python
take_profit = entry_price + (3 * atr[-1])  # Minimum 3x ATR for longs
take_profit = entry_price - (3 * atr[-1])  # Minimum 3x ATR for shorts
```

**Why:** ATR adapts to volatility. BTC at $20k vs $100k has different point values.

---

### 2. Multi-Timeframe Trend Filter ✅

**Always require higher TF confirmation:**
```python
# Example: 4H trend filter for 15m entries
def preprocess_data(df, **params):
    # Resample to 4H
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    # 4H EMA200
    df_4h['ema_200'] = df_4h['Close'].ewm(span=200).mean()
    
    # Forward fill to 15m
    df['htf_trend'] = (df_4h['Close'] > df_4h['ema_200']).reindex(df.index, method='ffill')
    
    return df

# In strategy:
if self.data.htf_trend[-1]:  # Only long in 4H uptrend
    self.buy()
```

---

### 3. Volume Confirmation ✅

**Require above-average volume:**
```python
volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)

# In entry logic:
if self.data.Volume[-1] > volume_ma[-1]:
    self.buy()  # High volume confirms move
```

---

### 4. Market Regime Detection ✅

**Classify and adapt:**
```python
def detect_regime(self):
    atr = talib.ATR(self.data.High, self.data.Low, self.data.Close, 14)
    atr_ma = np.mean(atr[-50:])
    
    # High vs low volatility
    high_vol = atr[-1] > atr_ma * 1.5
    
    # Trending vs ranging
    ema_50 = talib.EMA(self.data.Close, 50)
    ema_200 = talib.EMA(self.data.Close, 200)
    trending = abs(ema_50[-1] - ema_200[-1]) / self.data.Close[-1] > 0.02  # 2% separation
    
    if high_vol and trending:
        return "TRENDING_HIGH_VOL"
    elif high_vol and not trending:
        return "RANGING_HIGH_VOL"
    elif not high_vol and trending:
        return "TRENDING_LOW_VOL"
    else:
        return "RANGING_LOW_VOL"

# In strategy:
regime = self.detect_regime()
if regime == "TRENDING_HIGH_VOL":
    position_size = 1.0  # Normal size
elif regime == "RANGING_LOW_VOL":
    return  # Skip - low edge
```

---

### 5. Normalized Thresholds ✅

**Express everything relative to price or ATR:**

❌ **Bad:**
```python
if price_move > 5:  # Fixed points
```

✅ **Good:**
```python
if price_move > (0.5 * atr[-1]):  # Half ATR move
```

---

## Prohibited Patterns

### ❌ Hard-Coded Time Windows

**Bad:**
```python
if current_time.hour == 10:  # Only trade at 10 AM
```

**Acceptable Alternative:**
```python
# Use volatility-based session detection
if atr[-1] > atr_ma[-1] * 1.2:  # High volatility period
```

**Exception:** Session filters for specific markets (Forex London/NY) are OK if they're part of the strategy concept.

---

### ❌ Fixed Price Targets

**Bad:**
```python
take_profit = entry + 5  # Always +5 points
```

**Good:**
```python
take_profit = entry + (3 * atr[-1])  # 3x ATR
```

---

### ❌ Fixed Percentage Stops Without ATR

**Bad:**
```python
stop_loss = entry * 0.98  # Always 2% stop
```

**Good:**
```python
stop_loss = entry - (2 * atr[-1])  # 2x ATR stop
```

---

### ❌ Single Timeframe Analysis

**Bad:**
```python
# Entry based only on 15m chart
if ema_cross:
    self.buy()
```

**Good:**
```python
# 15m entry with 4H trend confirmation
if ema_cross and htf_uptrend:
    self.buy()
```

---

## Template Structure

All strategies should follow this structure:

```python
from backtesting import Strategy
import talib
import numpy as np
import pandas as pd

def preprocess_data(df, **params):
    """
    Add all indicators here.
    Include multi-timeframe features.
    """
    # ATR
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    
    # Higher timeframe trend
    df_4h = df.resample('4H').agg({...})
    df_4h['trend'] = ...
    df['htf_trend'] = df_4h['trend'].reindex(df.index, method='ffill')
    
    # Volume
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    
    return df

class MyStrategy(Strategy):
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_threshold = 1.0
    
    def init(self):
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.htf_trend = self.I(lambda: self.data.htf_trend, name='htf_trend')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
    
    def next(self):
        # 1. Check filters
        if not self.htf_trend[-1]:  # Wrong trend
            return
        
        if self.data.Volume[-1] < self.volume_ma[-1]:  # Low volume
            return
        
        # 2. Entry logic
        if self.entry_condition():
            sl = self.data.Close[-1] - (self.atr_sl_multiplier * self.atr[-1])
            tp = self.data.Close[-1] + (self.atr_tp_multiplier * self.atr[-1])
            self.buy(sl=sl, tp=tp)
```

---

## Validation Checklist

Before submitting a strategy PR, verify:

- [ ] Uses ATR for stop loss (2x minimum)
- [ ] Uses ATR for take profit (3x minimum)
- [ ] Includes higher timeframe trend filter
- [ ] Requires volume confirmation
- [ ] No hard-coded price targets
- [ ] No hard-coded time windows (unless strategy-specific)
- [ ] No fixed percentage stops
- [ ] Parameters are optimizable

---

## Red Team Validation

All strategies must pass:

1. **Stage 1:** 70/30 WFA split with <30% OOS degradation
2. **Stage 2:** Rolling Window WFA (5 windows)
3. **Stage 3:** Monte Carlo stress test (100 scenarios)

If a strategy fails Stage 1, review this document and refactor.

---

**Questions?** See `red_team_runner.py` for validation process.
