from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class BidAskVolumeStrategy(BaseStrategy):
    def __init__(self, window: int = 20, volume_threshold: float = 1.5, proximity: float = 0.15):
        super().__init__()
        self.window = window
        self.volume_threshold = volume_threshold
        self.proximity = proximity
        self.volumes = deque(maxlen=window)
        self.prices = deque(maxlen=2)
        
    def reset(self) -> None:
        self.volumes.clear()
        self.prices.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        spread = ask - bid
        if spread <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={'error': 'invalid_spread'})
            
        self.volumes.append(volume)
        self.prices.append(price)
        
        if len(self.volumes) < self.window:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'accumulating_data'})
            
        avg_volume = sum(self.volumes) / self.window
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio < self.volume_threshold:
            return Signal(
                action='HOLD',
                confidence=0.0,
                meta={
                    'avg_volume': avg_volume,
                    'current_volume': volume,
                    'reason': 'insufficient_volume'
                }
            )
            
        distance_from_bid = price - bid
        distance_from_ask = ask - price
        
        proximity_bid = distance_from_bid / spread
        proximity_ask = distance_from_ask / spread
        
        if proximity_bid < self.proximity:
            confidence = (1.0 - (proximity_bid / self.proximity)) * min(volume_ratio - 1.0, 1.0)
            confidence = max(0.0, min(confidence, 1.0))
            return Signal(
                action='BUY',
                confidence=confidence,
                meta={
                    'proximity_to_bid': proximity_bid,
                    'volume_ratio': volume_ratio,
                    'signal': 'support_with_volume'
                }
            )
        elif proximity_ask < self.proximity:
            confidence = (1.0 - (proximity_ask / self.proximity)) * min(volume_ratio - 1.0, 1.0)
            confidence = max(0.0, min(confidence, 1.0))
            return Signal(
                action='SELL',
                confidence=confidence,
                meta={
                    'proximity_to_ask': proximity_ask,
                    'volume_ratio': volume_ratio,
                    'signal': 'resistance_with_volume'
                }
            )
        else:
            return Signal(
                action='HOLD',
                confidence=0.0,
                meta={
                    'proximity_bid': proximity_bid,
                    'proximity_ask': proximity_ask,
                    'volume_ratio': volume_ratio
                }
            )