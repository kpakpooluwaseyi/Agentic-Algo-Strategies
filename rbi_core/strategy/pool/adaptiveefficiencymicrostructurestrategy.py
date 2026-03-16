from rbi_core.strategy.base import BaseStrategy
from collections import deque, namedtuple
from typing import Dict, Any
import math

Signal = namedtuple('Signal', ['action', 'confidence', 'meta'])

class AdaptiveEfficiencyMicrostructureStrategy(BaseStrategy):
    def __init__(self, lookback=20, efficiency_period=10, cooldown_ticks=8):
        super().__init__()
        self.lookback = lookback
        self.efficiency_period = efficiency_period
        self.cooldown_ticks = cooldown_ticks
        self.reset()
    
    def reset(self):
        self.prices = deque(maxlen=self.lookback)
        self.volumes = deque(maxlen=self.lookback)
        self.spreads_pct = deque(maxlen=10)
        self.atrs = deque(maxlen=5)
        self.price_deltas = deque(maxlen=self.efficiency_period)
        self.cooldown_counter = 0
        self.prev_price = None
        self.last_regime = 'neutral'
    
    def _calculate_efficiency_ratio(self, current_price):
        if len(self.prices) < self.efficiency_period or len(self.price_deltas) < self.efficiency_period - 1:
            return 0.5
        
        start_price = list(self.prices)[-self.efficiency_period]
        net_movement = abs(current_price - start_price)
        total_movement = sum(self.price_deltas)
        
        if total_movement == 0:
            return 0.0
        return min(net_movement / total_movement, 1.0)
    
    def _adaptive_moving_average(self, data, adaptive_factor):
        if not data:
            return 0
        n = len(data)
        if n < 3:
            return sum(data) / n
        
        ema = data[0]
        alpha = min(0.5, max(0.1, adaptive_factor / n))
        
        for price in list(data)[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def on_tick(self, tick_data: Dict[str, Any]) -> Signal:
        price = tick_data['price']
        volume = tick_data['volume']
        atr = tick_data['atr']
        bid = tick_data['bid']
        ask = tick_data['ask']
        
        spread = ask - bid
        spread_pct = spread / price if price > 0 else 0.0001
        
        if self.prev_price is not None:
            self.price_deltas.append(abs(price - self.prev_price))
        self.prev_price = price
        
        self.prices.append(price)
        self.volumes.append(volume)
        self.spreads_pct.append(spread_pct)
        self.atrs.append(atr)
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return Signal('HOLD', 0.0, {'reason': 'signal_cooldown'})
        
        if len(self.prices) < self.lookback:
            return Signal('HOLD', 0.0, {'reason': 'initializing'})
        
        avg_vol = sum(self.volumes) / len(self.volumes)
        rel_volume = volume / avg_vol if avg_vol > 0 else 1.0
        
        avg_spread = sum(self.spreads_pct) / len(self.spreads_pct)
        liquidity_score = max(0, 1