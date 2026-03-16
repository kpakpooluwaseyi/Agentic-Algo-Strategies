from typing import Optional
from rbi_core.strategy.base import BaseStrategy, Signal

class ATRMomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.prev_price = None
        self.volume_history = []
        self.lookback = 15
        
    def reset(self) -> None:
        self.prev_price = None
        self.volume_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        if self.prev_price is None or atr <= 0:
            self.prev_price = price
            self.volume_history.append(volume)
            if len(self.volume_history) > self.lookback:
                self.volume_history.pop(0)
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        price_change = price - self.prev_price
        normalized_change = price_change / atr
        
        avg_volume = sum(self.volume_history) / len(self.volume_history) if self.volume_history else volume
        volume_factor = volume / avg_volume if avg_volume > 0 else 1.0
        
        action = 'HOLD'
        confidence = 0.0
        meta = {
            'normalized_change': normalized_change,
            'volume_factor': volume_factor,
            'atr': atr
        }
        
        if normalized_change > 0.4 and volume_factor > 1.15:
            action = 'BUY'
            confidence = min(0.6 + abs(normalized_change) * 0.2 + (volume_factor - 1) * 0.2, 1.0)
        elif normalized_change < -0.4 and volume_factor > 1.15:
            action = 'SELL'
            confidence = min(0.6 + abs(normalized_change) * 0.2 + (volume_factor - 1) * 0.2, 1.0)
        
        self.prev_price = price
        self.volume_history.append(volume)
        if len(self.volume_history) > self.lookback:
            self.volume_history.pop(0)
        
        return Signal(action=action, confidence=confidence, meta=meta)