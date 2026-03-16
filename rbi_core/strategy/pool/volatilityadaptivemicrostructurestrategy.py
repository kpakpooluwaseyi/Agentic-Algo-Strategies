from rbi_core.strategy.base import BaseStrategy
from collections import deque
from typing import Dict, Any, NamedTuple
import math

class Signal(NamedTuple):
    action: str
    confidence: float
    meta: Dict[str, Any]

class VolatilityAdaptiveMicrostructureStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        cfg = config or {}
        self.lookback = cfg.get('lookback', 20)
        self.pressure_window = cfg.get('pressure_window', 5)
        self.momentum_threshold = cfg.get('momentum_threshold', 0.15)
        self.volume_surge_threshold = cfg.get('volume_surge_threshold', 1.4)
        self.atr_volatility_limit = cfg.get('atr_volatility_limit', 2.0)
        self.ema_alpha_price = cfg.get('ema_alpha_price', 0.15)
        self.ema_alpha_volume = cfg.get('ema_alpha_volume', 0.1)
        
        self.price_history = deque(maxlen=self.lookback)
        self.volume_history = deque(maxlen=self.lookback)
        self.atr_history = deque(maxlen=self.lookback)
        self.pressure_history = deque(maxlen=self.pressure_window)
        self.tick_count = 0
        
        self.ema_price = None
        self.ema_volume = None
        self.position_state = 'FLAT'
        self.entry_price = 0.0
        self.cumulative_pnl = 0.0
        
        self._welford_mean = 0.0
        self._welford_m2 = 0.0
        self._volatility_estimate = 0.0
        
    def reset(self):
        self.price_history.clear()
        self.volume_history.clear()
        self.atr_history.clear()
        self.pressure_history.clear()
        self.tick_count = 0
        self.ema_price = None
        self.ema_volume = None
        self.position_state = 'FLAT'
        self.entry_price = 0.0
        self.cumulative_pnl = 0.0
        self._welford_mean = 0.0
        self._welford_m2 = 0.0
        self._volatility_estimate = 0.0
        
    def _update_volatility_estimate(self, value: float):
        self.tick_count += 1
        delta = value - self._welford_mean
        self._welford_mean += delta / self.tick_count
        delta2 = value - self._welford_mean
        self._welford_m2 += delta * delta2
        if self.tick_count > 1:
            variance = self._welford_m2 / (self.tick_count - 1)
            self._volatility_estimate = math.sqrt(variance) if variance > 0 else 0.0
            
    def _calculate_microstructure_pressure(self, tick: Dict[str, Any]) -> float:
        bid = tick['bid']
        ask = tick['ask']
        price = tick['price']
        volume = tick['volume']
        
        if bid >= ask or ask == 0:
            return 0.0
            
        spread = ask - bid
        mid_price = (bid + ask) / 2.0
        position_in_spread = (price - bid) / spread if spread > 0 else 0.5
        
        bid_ask_imbalance = (ask - price) / spread if price > mid_price else -(price - bid) / spread
        bid_ask_imbalance = max(-1.0, min(1.0, bid_ask_imbalance))
        
        if len(self.volume_history) < 2:
            volume_z = 0.0
        else:
            mean_vol = sum(self.volume_history) / len(self.volume_history)
            if self.ema_volume and self.ema_volume > 0:
                volume_z = (volume - mean_vol) / self.ema_volume
            else:
                volume_z = 0.0
                
        micro_pressure = (position_in_spread * 0.35) + (bid_ask_imbalance * 0.35) + (math.tanh(volume_z) * 0.3)
        return max(-1.0, min(1.0, micro_pressure))
        
    def _calculate_adaptive_momentum(self, current_price: float) -> float:
        if len(self.price_history) < 5:
            return 0.0
            
        recent_prices = list(self.price_history)
        short_window = recent_prices[-5:]
        short_mean = sum(short_window) / len(short_window)
        
        long_mean = sum(recent_prices) / len(recent_prices)
        
        price_change_short = (current_price - short_mean) / short_mean if short_mean != 0 else 0
        price_change_long = (current_price - long_mean) / long_mean if long_mean != 0 else 0
        
        if self._volatility_estimate > 0 and self.ema_price and self.ema_price > 0:
            normalized_vol = self._volatility_estimate / self.ema_price
            vol_adjustment = 1.0 / (1.0 + normalized_vol * 10)
        else:
            vol_adjustment = 1.0
            
        raw_momentum = (price_change_short * 0.6 + price_change_long * 0.4) * vol_adjustment
        return math.tanh(raw_momentum * 10