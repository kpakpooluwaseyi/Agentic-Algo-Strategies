from typing import Optional
from collections import deque
import math
from rbi_core.strategy.base import BaseStrategy, Signal

class BollingerBandMeanReversion(BaseStrategy):
    def __init__(self):
        self.window = 20
        self.std_mult = 2.0
        self.price_history = deque(maxlen=self.window)
        self.volume_history = deque(maxlen=self.window)
        
    def reset(self) -> None:
        self.price_history.clear()
        self.volume_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        if len(self.price_history) < self.window:
            self.price_history.append(price)
            self.volume_history.append(volume)
            return None
            
        prices = list(self.price_history)
        volumes = list(self.volume_history)
        
        sma = sum(prices) / len(prices)
        variance = sum((p - sma) ** 2 for p in prices) / len(prices)
        std = math.sqrt(variance) if variance > 0 else 0.0
        
        upper_band = sma + self.std_mult * std
        lower_band = sma - self.std_mult * std
        avg_volume = sum(volumes) / len(volumes) if volumes else 0.0
        
        action = 'HOLD'
        confidence = 0.0
        meta = {'sma': sma, 'std': std}
        
        if price < lower_band and volume > avg_volume:
            action = 'BUY'
            deviation = (lower_band - price) / (std * self.std_mult) if std > 0 else 0
            confidence = min(1.0, 0.6 + deviation * 0.2)
        elif price > upper_band and volume > avg_volume:
            action = 'SELL'
            deviation = (price - upper_band) / (std * self.std_mult) if std > 0 else 0
            confidence = min(1.0, 0.6 + deviation * 0.2)
            
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        return Signal(action=action, confidence=confidence, meta=meta)