from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional, Deque
from collections import deque

class ATRVolatilityBreakoutStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, atr_multiplier: float = 1.5, volume_threshold: float = 1.2):
        super().__init__()
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier
        self.volume_threshold = volume_threshold
        self.prices: Deque[float] = deque(maxlen=lookback)
        self.volumes: Deque[float] = deque(maxlen=lookback)
        self.last_signal: str = 'HOLD'
        
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.last_signal = 'HOLD'
        
    def _calculate_sma(self, data: Deque[float]) -> float:
        if not data:
            return 0.0
        return sum(data) / len(data)
        
    def _calculate_volume_trend(self) -> float:
        if len(self.volumes) < 2:
            return 1.0
        recent = list(self.volumes)[-5:] if len(self.volumes) >= 5 else list(self.volumes)
        older = list(self.volumes)[:-len(recent)] if len(self.volumes) > len(recent) else []
        if not older:
            return 1.0
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        return avg_recent / avg_older if avg_older > 0 else 1.0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) < self.lookback:
            return None
            
        sma = self._calculate_sma(self.prices)
        upper_band = sma + (self.atr_multiplier * atr)
        lower_band = sma - (self.atr_multiplier * atr)
        volume_trend = self._calculate_volume_trend()
        
        if price > upper_band and volume_trend > self.volume_threshold and self.last_signal != 'BUY':
            self.last_signal = 'BUY'
            distance = (price - upper_band) / atr if atr > 0 else 0
            confidence = min(0.6 + (distance * 0.15) + (volume_trend - 1) * 0.15, 1.0)
            return Signal(action='BUY', confidence=confidence, meta={
                'sma': round(sma, 4),
                'upper_band': round(upper_band, 4),
                'volume_trend': round(volume_trend, 4),
                'atr': round(atr, 4)
            })
        elif price < lower_band and volume_trend > self.volume_threshold and self.last_signal != 'SELL':
            self.last_signal = 'SELL'
            distance = (lower_band - price) / atr if atr > 0 else 0
            confidence = min(0.6 + (distance * 0.15) + (volume_trend - 1) * 0.15, 1.0)
            return Signal(action='SELL', confidence=confidence, meta={
                'sma': round(sma, 4),
                'lower_band': round(lower_band, 4),
                'volume_trend': round(volume_trend, 4),
                'atr': round(atr, 4)
            })
            
        return None