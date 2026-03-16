from typing import Optional, List
from rbi_core.strategy.base import BaseStrategy, Signal


class ATRVolatilityBreakout(BaseStrategy):
    def __init__(self, lookback: int = 15, atr_factor: float = 1.2, vol_threshold: float = 1.1):
        self.lookback = lookback
        self.atr_factor = atr_factor
        self.vol_threshold = vol_threshold
        self.price_window: List[float] = []
        self.volume_window: List[float] = []
        self.prev_signal = 'HOLD'
        
    def reset(self) -> None:
        self.price_window.clear()
        self.volume_window.clear()
        self.prev_signal = 'HOLD'
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        self.price_window.append(price)
        self.volume_window.append(volume)
        
        if len(self.price_window) > self.lookback:
            self.price_window.pop(0)
            self.volume_window.pop(0)
            
        if len(self.price_window) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'state': 'warming_up'})
            
        avg_vol = sum(self.volume_window) / len(self.volume_window)
        highest = max(self.price_window)
        lowest = min(self.price_window)
        
        upper_band = highest + (atr * self.atr_factor)
        lower_band = lowest - (atr * self.atr_factor)
        
        if volume < avg_vol * self.vol_threshold:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'low_volume'})
            
        if price > upper_band and self.prev_signal != 'BUY':
            confidence = min(1.0, (price - upper_band) / (atr * 2) + 0.5)
            self.prev_signal = 'BUY'
            return Signal(action='BUY', confidence=confidence, meta={
                'trigger': 'upper_breakout',
                'band': upper_band,
                'atr': atr
            })
        elif price < lower_band and self.prev_signal != 'SELL':
            confidence = min(1.0, (lower_band - price) / (atr * 2) + 0.5)
            self.prev_signal = 'SELL'
            return Signal(action='SELL', confidence=confidence, meta={
                'trigger': 'lower_breakout', 
                'band': lower_band,
                'atr': atr
            })
            
        return Signal(action='HOLD', confidence=0.0, meta={
            'position': (price - lowest) / (highest - lowest) if highest != lowest else 0.5
        })