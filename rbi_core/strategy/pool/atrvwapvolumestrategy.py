from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional, Dict, Any
from collections import deque

class ATRVWAPVolumeStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, atr_multiplier: float = 1.5, volume_multiplier: float = 1.3):
        super().__init__()
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier
        self.volume_multiplier = volume_multiplier
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        current_price = tick_data.price
        current_volume = tick_data.volume
        current_atr = tick_data.atr
        
        self.prices.append(current_price)
        self.volumes.append(current_volume)
        
        if len(self.prices) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'accumulating_data'})
        
        total_pv = sum(p * v for p, v in zip(self.prices, self.volumes))
        total_vol = sum(self.volumes)
        if total_vol == 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        vwap = total_pv / total_vol
        avg_volume = total_vol / self.lookback
        
        upper_band = vwap + (self.atr_multiplier * current_atr)
        lower_band = vwap - (self.atr_multiplier * current_atr)
        
        action = 'HOLD'
        confidence = 0.0
        meta: Dict[str, Any] = {
            'vwap': vwap,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'avg_volume': avg_volume,
            'atr': current_atr
        }
        
        volume_threshold = avg_volume * self.volume_multiplier
        
        if current_price > upper_band and current_volume > volume_threshold:
            action = 'BUY'
            vol_excess = current_volume / volume_threshold if volume_threshold > 0 else 1.0
            confidence = min(1.0, 0.6 + (vol_excess - 1.0) * 0.2)
            meta['signal_type'] = 'breakout_upper'
        elif current_price < lower_band and current_volume > volume_threshold:
            action = 'SELL'
            vol_excess = current_volume / volume_threshold if volume_threshold > 0 else 1.0
            confidence = min(1.0, 0.6 + (vol_excess - 1.0) * 0.2)
            meta['signal_type'] = 'breakout_lower'
        
        return Signal(action=action, confidence=confidence, meta=meta)