from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class ATRChannelStrategy(BaseStrategy):
    def __init__(self):
        self.price_history = deque(maxlen=20)
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        
        if len(self.price_history) < 20:
            self.price_history.append(price)
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        highest_high = max(self.price_history)
        lowest_low = min(self.price_history)
        self.price_history.append(price)
        
        if price > highest_high + 0.5 * atr:
            return Signal(action='BUY', confidence=0.85, meta={'trigger': 'upper_breakout', 'channel_width': highest_high - lowest_low})
        elif price < lowest_low - 0.5 * atr:
            return Signal(action='SELL', confidence=0.85, meta={'trigger': 'lower_breakout', 'channel_width': highest_high - lowest_low})
        else:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
    def reset(self) -> None:
        self.price_history.clear()