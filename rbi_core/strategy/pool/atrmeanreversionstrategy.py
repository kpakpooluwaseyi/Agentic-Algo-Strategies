from typing import Optional
from collections import deque
from rbi_core.strategy.base import BaseStrategy, Signal

class ATRMeanReversionStrategy(BaseStrategy):
    def __init__(self):
        self.price_history = deque(maxlen=20)
        self.volume_history = deque(maxlen=20)
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if len(self.price_history) < 20:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        sma = sum(self.price_history) / len(self.price_history)
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        
        upper_band = sma + (1.5 * atr)
        lower_band = sma - (1.5 * atr)
        
        meta = {
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'avg_volume': avg_volume,
            'current_atr': atr
        }
        
        if price <= lower_band and volume > avg_volume * 1.2:
            deviation = (lower_band - price) / atr if atr > 0 else 0
            confidence = min(1.0, 0.5 + deviation * 0.5)
            return Signal(action='BUY', confidence=confidence, meta=meta)
        elif price >= upper_band and volume > avg_volume * 1.2:
            deviation = (price - upper_band) / atr if atr > 0 else 0
            confidence = min(1.0, 0.5 + deviation * 0.5)
            return Signal(action='SELL', confidence=confidence, meta=meta)
        
        return Signal(action='HOLD', confidence=0.0, meta=meta)
    
    def reset(self) -> None:
        self.price_history.clear()
        self.volume_history.clear()