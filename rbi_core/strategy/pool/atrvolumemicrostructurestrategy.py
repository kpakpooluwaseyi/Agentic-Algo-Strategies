from typing import Optional
from collections import deque
from rbi_core.strategy.base import BaseStrategy, Signal

class ATRVolumeMicrostructureStrategy(BaseStrategy):
    def __init__(self, lookback: int = 15):
        super().__init__()
        self.lookback = lookback
        self.price_history = deque(maxlen=lookback)
        self.volume_history = deque(maxlen=lookback)
        self.last_atr = 0.0
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        
    def reset(self) -> None:
        self.price_history.clear()
        self.volume_history.clear()
        self.last_atr = 0.0
        self.position = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        bid = tick_data.bid
        ask = tick_data.ask
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.last_atr = atr
        
        if len(self.price_history) < self.lookback:
            return None
            
        avg_price = sum(self.price_history) / len(self.price_history)
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        
        recent_high = max(self.price_history)
        recent_low = min(self.price_history)
        
        spread = ask - bid
        micro_pressure = (price - bid) / spread if spread > 0 else 0.5
        
        volatility_state = "high" if atr > avg_price * 0.002 else "low"
        
        signal = None
        
        if self.position <= 0:
            breakout_threshold = recent_high - (atr * 0.2)
            if price >= breakout_threshold and volume > avg_volume * 1.15 and micro_pressure > 0.65:
                confidence = min(0.95, 0.55 + (micro_pressure - 0.5) * 0.6 + (volume / avg_volume - 1) * 0.3)
                signal = Signal(
                    action='BUY',
                    confidence=confidence,
                    meta={
                        'type': 'volatility_breakout',
                        'atr': atr,
                        'micro_pressure': micro_pressure,
                        'volume_ratio': volume / avg_volume,
                        'volatility_state': volatility_state
                    }
                )
                self.position = 1
                
        if self.position >= 0:
            breakdown_threshold = recent_low + (atr * 0.2)
            if price <= breakdown_threshold and volume > avg_volume * 1.15 and micro_pressure < 0.35:
                confidence = min(0.95, 0.55 + (0.5 - micro_pressure) * 0.6 + (volume / avg_volume - 1) * 0.3)
                signal = Signal(
                    action='SELL',
                    confidence=confidence,
                    meta={
                        'type': 'volatility_breakdown',
                        'atr': atr,
                        'micro_pressure': micro_pressure,
                        'volume_ratio': volume / avg_volume,
                        'volatility_state': volatility_state
                    }
                )
                self.position = -1
                
        return signal