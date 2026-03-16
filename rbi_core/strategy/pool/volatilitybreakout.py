from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class VolatilityBreakout(BaseStrategy):
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        self.prices = deque(maxlen=20)
        self.volumes = deque(maxlen=20)
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) < 20:
            return None
            
        highest = max(self.prices)
        lowest = min(self.prices)
        avg_volume = sum(self.volumes) / len(self.volumes)
        
        breakout_buffer = 0.2 * atr
        
        if price > highest - breakout_buffer and volume > avg_volume * 1.2:
            confidence = min(0.5 + (price - (highest - breakout_buffer)) / (atr + 1e-9), 1.0)
            return Signal(action='BUY', confidence=confidence,
                         meta={'channel_high': highest, 'volume_factor': volume/avg_volume})
                         
        if price < lowest + breakout_buffer and volume > avg_volume * 1.2:
            confidence = min(0.5 + ((lowest + breakout_buffer) - price) / (atr + 1e-9), 1.0)
            return Signal(action='SELL', confidence=confidence,
                         meta={'channel_low': lowest, 'volume_factor': volume/avg_volume})
        
        return Signal(action='HOLD', confidence=0.0, meta={})