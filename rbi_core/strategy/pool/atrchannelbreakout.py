from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class AtrChannelBreakout(BaseStrategy):
    """
    A volatility breakout strategy using ATR to define dynamic channels.
    Generates a BUY signal when price breaks above the Upper Channel.
    Generates a SELL signal when price breaks below the Lower Channel.
    """
    def __init__(self, window_size: int = 20, atr_multiplier: float = 2.0):
        super().__init__()
        self.window_size = window_size
        self.atr_multiplier = atr_multiplier
        self.price_history: deque = deque(maxlen=window_size)
        self.last_signal_action: str = 'HOLD'

    def reset(self) -> None:
        self.price_history.clear()
        self.last_signal_action = 'HOLD'

    def on_tick(self, tick_data) -> Optional[Signal]:
        self.price_history.append(tick_data.price)
        
        if len(self.price_history) < self.window_size or tick_data.atr is None:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'accumulating_history'})

        highest_high = max(self.price_history)
        lowest_low = min(self.price_history)
        
        upper_band = highest_high + (tick_data.atr * self.atr_multiplier)
        lower_band = lowest_low - (tick_data.atr * self.atr_multiplier)
        
        current_price = tick_data.price
        
        action = 'HOLD'
        confidence = 0.0
        meta = {
            'upper_band': upper_band,
            'lower_band': lower_band,
            'atr': tick_data.atr
        }

        if current_price > upper_band and self.last_signal_action != 'BUY':
            action = 'BUY'
            # Confidence scales with how far price exceeds the band, capped at 1.0
            raw_strength = (current_price - upper_band) / tick_data.atr if tick_data.atr > 0 else 0
            confidence = min(1.0, 0.5 + raw_strength * 0.1)
            self.last_signal_action = 'BUY'
        elif current_price < lower_band and self.last_signal_action != 'SELL':
            action = 'SELL'
            raw_strength = (lower_band - current_price) / tick_data.atr if tick_data.atr > 0 else 0
            confidence = min(1.0, 0.5 + raw_strength * 0.1)
            self.last_signal_action = 'SELL'
        
        return Signal(action=action, confidence=confidence, meta=meta)