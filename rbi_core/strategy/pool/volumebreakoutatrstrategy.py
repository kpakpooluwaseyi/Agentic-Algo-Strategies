from typing import Optional, Deque
from collections import deque
from rbi_core.strategy.base import BaseStrategy, Signal

class VolumeBreakoutATRStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, volume_mult: float = 1.5, min_atr: float = 0.0, max_spread: float = 0.001):
        super().__init__()
        self.lookback = lookback
        self.volume_mult = volume_mult
        self.min_atr = min_atr
        self.max_spread = max_spread
        self.price_history: Deque[float] = deque(maxlen=lookback)
        self.volume_history: Deque[float] = deque(maxlen=lookback)
        
    def reset(self) -> None:
        self.price_history.clear()
        self.volume_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        bid = tick_data.bid
        ask = tick_data.ask
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if len(self.price_history) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'initializing', 'samples': len(self.price_history)})
            
        price_sma = sum(self.price_history) / self.lookback
        vol_sma = sum(self.volume_history) / self.lookback
        
        spread = (ask - bid) / price if price > 0 else float('inf')
        
        if atr < self.min_atr:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'low_atr', 'atr': atr})
            
        if spread > self.max_spread:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'wide_spread', 'spread': spread})
            
        if vol_sma == 0:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'zero_volume_sma'})
            
        vol_ratio = volume / vol_sma
        price_dev = (price - price_sma) / price_sma if price_sma != 0 else 0
        
        action = 'HOLD'
        confidence = 0.0
        
        if vol_ratio > self.volume_mult:
            if price_dev > 0.001:
                action = 'BUY'
                confidence = min(0.5 + (vol_ratio - self.volume_mult) * 0.2 + abs(price_dev) * 10, 1.0)
            elif price_dev < -0.001:
                action = 'SELL'
                confidence = min(0.5 + (vol_ratio - self.volume_mult) * 0.2 + abs(price_dev) * 10, 1.0)
                
        meta = {
            'price_sma': price_sma,
            'vol_sma': vol_sma,
            'vol_ratio': vol_ratio,
            'price_deviation': price_dev,
            'atr': atr,
            'spread': spread
        }
        
        return Signal(action=action, confidence=confidence, meta=meta)