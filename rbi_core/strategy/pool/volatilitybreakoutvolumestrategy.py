from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class VolatilityBreakoutVolumeStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, volume_mult: float = 1.5, atr_mult: float = 0.5):
        super().__init__()
        self.lookback = lookback
        self.volume_mult = volume_mult
        self.atr_mult = atr_mult
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        self.last_signal = 'HOLD'
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        bid = tick_data.bid
        ask = tick_data.ask
        
        spread_pct = (ask - bid) / price if price > 0 else 0
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'initializing'})
        
        avg_volume = sum(self.volumes) / self.lookback
        avg_price = sum(self.prices) / self.lookback
        
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        deviation = abs(price - avg_price)
        volatility_threshold = atr * self.atr_mult
        
        action = 'HOLD'
        confidence = 0.0
        meta = {
            'avg_price': avg_price,
            'volume_ratio': volume_ratio,
            'deviation': deviation,
            'atr': atr,
            'spread_pct': spread_pct
        }
        
        if volume_ratio >= self.volume_mult and deviation >= volatility_threshold:
            if price > avg_price:
                action = 'BUY'
                momentum = (price - avg_price) / avg_price if avg_price > 0 else 0
                confidence = min(0.6 + 0.3 * (volume_ratio - self.volume_mult) + 0.1 * (momentum / (atr/price)), 1.0)
            else:
                action = 'SELL'
                momentum = (avg_price - price) / avg_price if avg_price > 0 else 0
                confidence = min(0.6 + 0.3 * (volume_ratio - self.volume_mult) + 0.1 * (momentum / (atr/price)), 1.0)
        
        self.last_signal = action
        meta['confidence'] = confidence
        return Signal(action=action, confidence=confidence, meta=meta)
    
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.last_signal = 'HOLD'