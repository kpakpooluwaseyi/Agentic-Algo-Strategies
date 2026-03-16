from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque
import math

class KeltnerAtrBreakout(BaseStrategy):
    def __init__(self, window: int = 20, atr_multiplier: float = 2.0):
        super().__init__()
        self.window = window
        self.atr_multiplier = atr_multiplier
        self.prices = deque(maxlen=window)
        self.prev_position = 0  # 0: flat, 1: long, -1: short
        
    def reset(self) -> None:
        self.prices.clear()
        self.prev_position = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        
        self.prices.append(price)
        
        if len(self.prices) < self.window:
            return None
            
        sma = sum(self.prices) / len(self.prices)
        upper_band = sma + self.atr_multiplier * atr
        lower_band = sma - self.atr_multiplier * atr
        
        action = 'HOLD'
        confidence = 0.0
        meta = {
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'atr': atr,
            'price': price
        }
        
        # Trend following breakout logic
        if price > upper_band and self.prev_position <= 0:
            action = 'BUY'
            deviation = (price - upper_band) / atr if atr > 0 else 0
            confidence = min(0.95, 0.55 + deviation * 0.15)
            self.prev_position = 1
        elif price < lower_band and self.prev_position >= 0:
            action = 'SELL'
            deviation = (lower_band - price) / atr if atr > 0 else 0
            confidence = min(0.95, 0.55 + deviation * 0.15)
            self.prev_position = -1
            
        if action == 'HOLD':
            return None
            
        meta['deviation'] = deviation
        return Signal(action=action, confidence=confidence, meta=meta)