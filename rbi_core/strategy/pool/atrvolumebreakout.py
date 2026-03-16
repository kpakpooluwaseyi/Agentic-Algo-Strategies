from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque
import math


class ATRVolumeBreakout(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.price_history = deque(maxlen=20)
        self.volume_history = deque(maxlen=20)
        self.atr_multiplier = 1.5
        self.volume_boost = 1.3
        
    def reset(self) -> None:
        self.price_history.clear()
        self.volume_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        if len(self.price_history) < self.price_history.maxlen:
            self.price_history.append(price)
            self.volume_history.append(volume)
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        sma = sum(self.price_history) / len(self.price_history)
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        upper_band = sma + (atr * self.atr_multiplier)
        lower_band = sma - (atr * self.atr_multiplier)
        
        if volume > avg_volume * self.volume_boost:
            if price > upper_band:
                return Signal(
                    action='BUY', 
                    confidence=0.85, 
                    meta={'sma': sma, 'upper': upper_band, 'atr': atr}
                )
            elif price < lower_band:
                return Signal(
                    action='SELL', 
                    confidence=0.85, 
                    meta={'sma': sma, 'lower': lower_band, 'atr': atr}
                )
        
        return Signal(action='HOLD', confidence=0.0, meta={'sma': sma})