from typing import Optional, List
from collections import deque
from rbi_core.strategy.base import BaseStrategy, Signal

class ATRTrendStrategy(BaseStrategy):
    def __init__(self, period: int = 20, atr_multiplier: float = 1.5):
        super().__init__()
        self.period = period
        self.atr_multiplier = atr_multiplier
        self.prices: deque = deque(maxlen=period)
    
    def reset(self) -> None:
        self.prices.clear()
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        self.prices.append(price)
        
        if len(self.prices) < self.period:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        sma = sum(self.prices) / len(self.prices)
        upper_band = sma + self.atr_multiplier * atr
        lower_band = sma - self.atr_multiplier * atr
        
        if price > upper_band:
            distance = (price - upper_band) / atr if atr > 0 else 0
            confidence = min(1.0, 0.5 + distance * 0.25)
            return Signal(action='BUY', confidence=confidence, meta={'sma': sma, 'upper': upper_band})
        elif price < lower_band:
            distance = (lower_band - price) / atr if atr > 0 else 0
            confidence = min(1.0, 0.5 + distance * 0.25)
            return Signal(action='SELL', confidence=confidence, meta={'sma': sma, 'lower': lower_band})
        return Signal(action='HOLD', confidence=0.0, meta={'sma': sma})