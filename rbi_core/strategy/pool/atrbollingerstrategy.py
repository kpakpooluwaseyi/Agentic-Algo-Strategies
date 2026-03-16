from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class ATRBollingerStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.price_history = deque(maxlen=20)
        self.atr_multiplier = 2.0
        
    def reset(self) -> None:
        self.price_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        self.price_history.append(tick_data.price)
        
        if len(self.price_history) < 20:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        sma = sum(self.price_history) / len(self.price_history)
        atr = tick_data.atr
        
        upper_band = sma + (self.atr_multiplier * atr)
        lower_band = sma - (self.atr_multiplier * atr)
        
        if tick_data.price >= upper_band:
            deviation = (tick_data.price - upper_band) / atr if atr > 0 else 0
            confidence = min(1.0, 0.5 + deviation)
            return Signal(action='SELL', confidence=confidence, meta={'sma': sma, 'band': 'upper'})
        elif tick_data.price <= lower_band:
            deviation = (lower_band - tick_data.price) / atr if atr > 0 else 0
            confidence = min(1.0, 0.5 + deviation)
            return Signal(action='BUY', confidence=confidence, meta={'sma': sma, 'band': 'lower'})
        return Signal(action='HOLD', confidence=0.0, meta={'sma': sma, 'band': 'middle'})