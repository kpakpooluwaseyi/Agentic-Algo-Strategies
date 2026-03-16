from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class VWAPMeanReversion(BaseStrategy):
    def __init__(self, window: int = 50, deviation_threshold: float = 0.02, volume_multiplier: float = 1.5):
        self.window = window
        self.deviation_threshold = deviation_threshold
        self.volume_multiplier = volume_multiplier
        self.data = deque(maxlen=window)
        self.cooldown = 0
        
    def reset(self) -> None:
        self.data.clear()
        self.cooldown = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        if self.cooldown > 0:
            self.cooldown -= 1
            
        price = tick_data.price
        volume = tick_data.volume
        
        self.data.append((price, volume))
        
        if len(self.data) < self.window:
            return None
            
        total_pv = sum(p * v for p, v in self.data)
        total_v = sum(v for _, v in self.data)
        
        if total_v == 0:
            return None
            
        vwap = total_pv / total_v
        avg_volume = total_v / len(self.data)
        
        deviation = (price - vwap) / vwap if vwap != 0 else 0
        
        if abs(deviation) > self.deviation_threshold and volume > avg_volume * self.volume_multiplier and self.cooldown == 0:
            self.cooldown = 10
            
            if deviation < 0:
                return Signal(
                    action='BUY',
                    confidence=min(0.95, abs(deviation) * 10 + 0.5),
                    meta={'vwap': vwap, 'deviation': deviation, 'volume_ratio': volume / avg_volume}
                )
            else:
                return Signal(
                    action='SELL',
                    confidence=min(0.95, abs(deviation) * 10 + 0.5),
                    meta={'vwap': vwap, 'deviation': deviation, 'volume_ratio': volume / avg_volume}
                )
                
        return None

# ===