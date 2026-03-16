from rbi_core.strategy.base import BaseStrategy, Signal
from collections import deque
from typing import Optional

class VolumeConfirmedATRBreakout(BaseStrategy):
    def __init__(self, window: int = 20, atr_multiplier: float = 1.0, volume_multiplier: float = 1.5):
        self.window = window
        self.atr_multiplier = atr_multiplier
        self.volume_multiplier = volume_multiplier
        self.prices = deque(maxlen=window)
        self.volumes = deque(maxlen=window)
    
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) < self.window:
            return Signal(action='HOLD', confidence=0.0, meta={'warmup': len(self.prices)})
        
        avg_price = sum(self.prices) / self.window
        avg_volume = sum(self.volumes) / self.window
        
        upper_band = avg_price + self.atr_multiplier * atr
        lower_band = avg_price - self.atr_multiplier * atr
        
        meta = {
            'sma': avg_price,
            'avg_vol': avg_volume,
            'upper': upper_band,
            'lower': lower_band
        }
        
        if price > upper_band and volume > avg_volume * self.volume_multiplier:
            price_dev = (price - upper_band) / (atr + 1e-9)
            vol_ratio = volume / (avg_volume + 1e-9)
            confidence = min(1.0, (price_dev * 0.6 + (vol_ratio - 1) * 0.4))
            return Signal(action='BUY', confidence=confidence, meta=meta)
        elif price < lower_band and volume > avg_volume * self.volume_multiplier:
            price_dev = (lower_band - price) / (atr + 1e-9)
            vol_ratio = volume / (avg_volume + 1e-9)
            confidence = min(1.0, (price_dev * 0.6 + (vol_ratio - 1) * 0.4))
            return Signal(action='SELL', confidence=confidence, meta=meta)
        else:
            return Signal(action='HOLD', confidence=0.0, meta=meta)