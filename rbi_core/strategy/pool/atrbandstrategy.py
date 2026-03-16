from typing import Optional, Deque, Tuple
from collections import deque
from rbi_core.strategy.base import BaseStrategy, Signal
import math

class ATRBandStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, atr_multiplier: float = 1.5):
        super().__init__()
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier
        self.price_history: Deque[float] = deque(maxlen=lookback)
        self.current_position: int = 0
        
    def reset(self) -> None:
        self.price_history.clear()
        self.current_position = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        
        if len(self.price_history) < self.lookback:
            self.price_history.append(price)
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'initializing'})
        
        sma = sum(self.price_history) / len(self.price_history)
        upper_band = sma + (self.atr_multiplier * atr)
        lower_band = sma - (self.atr_multiplier * atr)
        
        action = 'HOLD'
        confidence = 0.0
        
        if price > upper_band and self.current_position <= 0:
            action = 'BUY'
            penetration = (price - upper_band) / atr if atr > 0 else 0
            confidence = min(1.0, 0.5 + penetration)
            self.current_position = 1
        elif price < lower_band and self.current_position >= 0:
            action = 'SELL'
            penetration = (lower_band - price) / atr if atr > 0 else 0
            confidence = min(1.0, 0.5 + penetration)
            self.current_position = -1
            
        self.price_history.append(price)
        
        return Signal(action=action, confidence=confidence, meta={
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'atr': atr,
            'position': self.current_position
        })