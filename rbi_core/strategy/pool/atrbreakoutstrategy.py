from typing import Optional
from collections import deque
import statistics
from rbi_core.strategy.base import BaseStrategy, Signal

class ATRBreakoutStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, atr_multiplier: float = 1.5):
        super().__init__()
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier
        self.price_history = deque(maxlen=lookback)
        
    def reset(self) -> None:
        self.price_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        
        if len(self.price_history) == self.lookback:
            highest_high = max(self.price_history)
            lowest_low = min(self.price_history)
            
            upper_trigger = highest_high + (atr * self.atr_multiplier)
            lower_trigger = lowest_low - (atr * self.atr_multiplier)
            
            if price > upper_trigger:
                distance = (price - upper_trigger) / (atr * 2) if atr != 0 else 0
                confidence = min(0.95, 0.5 + distance)
                return Signal(action='BUY', confidence=confidence, 
                            meta={'trigger': upper_trigger, 'level': 'breakout_high'})
            elif price < lower_trigger:
                distance = (lower_trigger - price) / (atr * 2) if atr != 0 else 0
                confidence = min(0.95, 0.5 + distance)
                return Signal(action='SELL', confidence=confidence,
                            meta={'trigger': lower_trigger, 'level': 'breakout_low'})
        
        self.price_history.append(price)
        return Signal(action='HOLD', confidence=0.0, meta={})