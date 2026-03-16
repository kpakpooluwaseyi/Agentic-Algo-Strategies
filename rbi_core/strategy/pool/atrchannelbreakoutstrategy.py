from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional, Deque
from collections import deque
from statistics import mean
import math


class ATRChannelBreakoutStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, atr_multiplier: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier
        self.price_history: Deque[float] = deque(maxlen=lookback)
        self.atr_history: Deque[float] = deque(maxlen=lookback)
        
    def reset(self) -> None:
        self.price_history.clear()
        self.atr_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        
        self.price_history.append(price)
        self.atr_history.append(atr)
        
        if len(self.price_history) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'initializing'})
            
        recent_high = max(self.price_history)
        recent_low = min(self.price_history)
        avg_atr = mean(self.atr_history)
        
        upper_band = recent_high + (avg_atr * self.atr_multiplier)
        lower_band = recent_low - (avg_atr * self.atr_multiplier)
        
        if price > upper_band:
            deviation = price - upper_band
            confidence = min(1.0, deviation / (avg_atr * 2)) if avg_atr > 0 else 0.5
            return Signal(
                action='BUY',
                confidence=confidence,
                meta={
                    'trigger': 'upper_breakout',
                    'band': upper_band,
                    'recent_high': recent_high,
                    'atr': atr
                }
            )
        elif price < lower_band:
            deviation = lower_band - price
            confidence = min(1.0, deviation / (avg_atr * 2)) if avg_atr > 0 else 0.5
            return Signal(
                action='SELL',
                confidence=confidence,
                meta={
                    'trigger': 'lower_breakout',
                    'band': lower_band,
                    'recent_low': recent_low,
                    'atr': atr
                }
            )
        else:
            position_in_range = (price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0.5
            return Signal(action='HOLD', confidence=0.0, meta={'position_in_range': position_in_range})