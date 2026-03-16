from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional, List
import statistics

class ATRMomentumBreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.max_lookback = 20
        self.breakout_threshold = 0.015
        self.atr_multiplier = 1.2
        
    def reset(self) -> None:
        self.price_history.clear()
        self.volume_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if len(self.price_history) > self.max_lookback:
            self.price_history.pop(0)
            self.volume_history.pop(0)
            
        if len(self.price_history) < 10:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'insufficient_history'})
        
        avg_volume = statistics.mean(self.volume_history[:-1]) if len(self.volume_history) > 1 else volume
        volume_surge = volume / avg_volume if avg_volume > 0 else 1.0
        
        recent_high = max(self.price_history[:-5]) if len(self.price_history) > 5 else max(self.price_history)
        recent_low = min(self.price_history[:-5]) if len(self.price_history) > 5 else min(self.price_history)
        
        range_size = recent_high - recent_low
        atr_threshold = atr * self.atr_multiplier
        
        meta = {
            'volume_surge': volume_surge,
            'range_size': range_size,
            'atr': atr
        }
        
        if range_size > atr_threshold and volume_surge > 1.5:
            if price > recent_high - (range_size * 0.1):
                confidence = min(0.95, 0.6 + (volume_surge - 1.5) * 0.2)
                return Signal(action='BUY', confidence=confidence, meta=meta)
            elif price < recent_low + (range_size * 0.1):
                confidence = min(0.95, 0.6 + (volume_surge - 1.5) * 0.2)
                return Signal(action='SELL', confidence=confidence, meta=meta)
                
        return Signal(action='HOLD', confidence=0.0, meta=meta)