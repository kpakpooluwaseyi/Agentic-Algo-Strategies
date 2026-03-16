from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class BollingerBandsReversion(BaseStrategy):
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__()
        self.window = window
        self.num_std = num_std
        self.price_history = deque(maxlen=window)
    
    def reset(self) -> None:
        self.price_history.clear()
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        self.price_history.append(price)
        
        if len(self.price_history) < self.window:
            return None
        
        mean = sum(self.price_history) / self.window
        variance = sum((p - mean) ** 2 for p in self.price_history) / self.window
        std_dev = variance ** 0.5
        
        upper = mean + self.num_std * std_dev
        lower = mean - self.num_std * std_dev
        
        if price < lower:
            confidence = min(1.0, (lower - price) / (std_dev * self.num_std)) if std_dev > 0 else 0.5
            return Signal(action='BUY', confidence=confidence, meta={
                'mean': mean, 'upper': upper, 'lower': lower, 'std': std_dev
            })
        elif price > upper:
            confidence = min(1.0, (price - upper) / (std_dev * self.num_std)) if std_dev > 0 else 0.5
            return Signal(action='SELL', confidence=confidence, meta={
                'mean': mean, 'upper': upper, 'lower': lower, 'std': std_dev
            })
        
        return None