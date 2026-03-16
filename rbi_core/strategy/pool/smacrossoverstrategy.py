from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque
import math

class SmaCrossoverStrategy(BaseStrategy):
    def __init__(self, fast_window: int = 10, slow_window: int = 30):
        super().__init__()
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.fast_prices = deque(maxlen=fast_window)
        self.slow_prices = deque(maxlen=slow_window)
        self.prev_fast_sma = None
        self.prev_slow_sma = None
    
    def reset(self) -> None:
        self.fast_prices.clear()
        self.slow_prices.clear()
        self.prev_fast_sma = None
        self.prev_slow_sma = None
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        self.fast_prices.append(price)
        self.slow_prices.append(price)
        
        if len(self.slow_prices) < self.slow_window:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        fast_sma = sum(self.fast_prices) / len(self.fast_prices)
        slow_sma = sum(self.slow_prices) / len(self.slow_prices)
        
        action = 'HOLD'
        confidence = 0.0
        
        if self.prev_fast_sma is not None and self.prev_slow_sma is not None:
            if self.prev_fast_sma <= self.prev_slow_sma and fast_sma > slow_sma:
                action = 'BUY'
                confidence = min(1.0, abs(fast_sma - slow_sma) / price * 100)
            elif self.prev_fast_sma >= self.prev_slow_sma and fast_sma < slow_sma:
                action = 'SELL'
                confidence = min(1.0, abs(fast_sma - slow_sma) / price * 100)
        
        self.prev_fast_sma = fast_sma
        self.prev_slow_sma = slow_sma
        
        return Signal(action=action, confidence=confidence, meta={'fast_sma': fast_sma, 'slow_sma': slow_sma})