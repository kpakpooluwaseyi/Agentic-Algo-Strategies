from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque
import math

class ATRVolumeBreakoutStrategy(BaseStrategy):
    def __init__(self):
        self.price_window = deque(maxlen=20)
        self.volume_window = deque(maxlen=20)
        self.atr_multiplier = 2.0
        
    def reset(self) -> None:
        self.price_window.clear()
        self.volume_window.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        self.price_window.append(price)
        self.volume_window.append(volume)
        
        if len(self.price_window) < 20:
            return Signal(action='HOLD', confidence=1.0, meta={})
        
        avg_price = sum(self.price_window) / len(self.price_window)
        avg_volume = sum(self.volume_window) / len(self.volume_window)
        
        deviation = abs(price - avg_price)
        threshold = atr * self.atr_multiplier
        
        if deviation > threshold and volume > avg_volume * 1.5:
            if price > avg_price:
                confidence = min(0.5 + (deviation / threshold - 1) * 0.25, 0.95)
                return Signal(action='BUY', confidence=confidence, meta={'indicator': 'atr_breakout', 'direction': 'up'})
            else:
                confidence = min(0.5 + (deviation / threshold - 1) * 0.25, 0.95)
                return Signal(action='SELL', confidence=confidence, meta={'indicator': 'atr_breakout', 'direction': 'down'})
        
        return Signal(action='HOLD', confidence=1.0, meta={})