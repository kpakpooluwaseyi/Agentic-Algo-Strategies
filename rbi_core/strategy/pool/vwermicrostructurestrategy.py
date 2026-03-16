from collections import deque
from dataclasses import dataclass
from typing import Dict, Any
from rbi_core.strategy.base import BaseStrategy

@dataclass
class Signal:
    action: str
    confidence: float
    meta: Dict[str, Any]

class VWERMicrostructureStrategy(BaseStrategy):
    def __init__(self, efficiency_lookback: int = 20, pressure_lookback: int = 10, 
                 cooldown_ticks: int = 5, min_atr: float = 0.0001):
        super().__init__()
        self.efficiency_lookback = efficiency_lookback
        self.pressure_lookback = pressure_lookback
        self.cooldown_ticks = cooldown_ticks
        self.min_atr = min_atr
        
        self.price_history = deque(maxlen=efficiency_lookback)
        self.volume_history = deque(maxlen=efficiency_lookback)
        self.atr_history = deque(maxlen=efficiency_lookback)
        self.pressure_history = deque(maxlen=pressure_lookback)
        
        self.tick_count = 0
        self.last_signal_tick = -cooldown_ticks
        self.prev_price = None
        
    def reset(self):
        self.price_history.clear()
        self.volume_history.clear()
        self.atr_history.clear()
        self.pressure_history.clear()
        self.tick_count = 0
        self.last_signal_tick = -self.cooldown_ticks
        self.prev_price = None
        
    def on_tick(self, tick_data: Dict[str, Any]) -> Signal:
        price = tick_data['price']
        volume = tick_data['volume']
        timestamp = tick_data['timestamp']
        atr = max(tick_data['atr'], self.min_atr)
        bid = tick_data['bid']
        ask = tick_data['ask']
        
        self.tick_count += 1
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.atr_history.append(atr)
        
        mid = (bid + ask) / 2.0
        spread = max(ask - bid, self.min_atr)
        imbalance = (bid - ask) / spread
        micro_pressure = imbalance * volume
        self.pressure_history.append(micro_pressure)
        
        meta = {
            'timestamp': timestamp,
            'tick': self.tick_count,
            'mid_price': mid,
            'spread': spread
        }
        
        if len(self.price_history) < self.efficiency_lookback:
            meta['state'] = 'warming_up'
            return Signal(action='HOLD', confidence=0.0, meta=meta)
        
        if len(self.pressure_history) < self.pressure_lookback:
            meta['state'] = 'pressure_init'
            return Signal(action='HOLD', confidence=0.0, meta=meta)
        
        net_movement = abs(price - self.price_history[0])
        total_volatility = sum(abs(self.price_history[i