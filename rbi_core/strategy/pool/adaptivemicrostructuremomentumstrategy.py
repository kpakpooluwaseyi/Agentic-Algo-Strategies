from rbi_core.strategy.base import BaseStrategy
from rbi_core.signal import Signal
from collections import deque
import math
from typing import Dict, Any

class AdaptiveMicrostructureMomentumStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, atr_period: int = 14, flow_sensitivity: float = 2.0):
        super().__init__()
        self.lookback = lookback
        self.atr_period = atr_period
        self.flow_sensitivity = flow_sensitivity
        
        # Circular buffers for state management
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        self.atrs = deque(maxlen=lookback)
        self.spreads = deque(maxlen=lookback)
        self.mid_prices = deque(maxlen=lookback)
        
        # Indicator state
        self.vwap_num = 0.0
        self.vwap_den = 0.0
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.ema_flow = 0.0
        self.atr_smoothed = 0.0
        
        # Adaptive parameters
        self.alpha_fast = 2.0 / (5.0 + 1.0)
        self.alpha_slow = 2.0 / (20.0 + 1.0)
        self.alpha_flow = 2.0 / (10.0 + 1.0)
        self.alpha_atr = 2.0 / (atr_period + 1.0)
        
    def reset(self):
        """Reset all stateful components for strategy reuse"""
        self.prices.clear()
        self.volumes.clear()
        self.atrs.clear()
        self.spreads.clear()
        self.mid_prices.clear()
        
        self.vwap_num = 0.0
        self.vwap_den = 0.0
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.ema_flow = 0.0
        self.atr_smoothed = 0.0
        
    def _calculate_microstructure_flow(self, bid: float, ask: float, price: float, volume: float) -> float:
        """Calculate order flow intensity based on position within spread and volume"""
        spread = ask - bid
        if spread <= 0 or volume <= 0:
            return 0.0
            
        # Normalized position in spread: -1 (at bid) to +1 (at ask)
        mid = (bid + ask) / 2.0
        position = (price - mid) / (spread / 2.0)
        position = max(-1.0, min(1.0, position))
        
        # Volume-weighted flow with logarithmic scaling to handle outliers
        flow = position * math.log1p(volume) * self.flow_sensitivity
        
        return math.tanh(flow)  # Bounded to [-1, 1]
        
    def _calculate_price_efficiency(self) -> float:
        """Calculate Kaufman-like efficiency ratio: net change / total movement"""
        if len(self.prices) < self.lookback:
            return 1.0
            
        prices_list = list(self.prices)
        net_change = abs(prices_list[-1] - prices_list[0])
        
        total_movement = sum(abs(prices_list[i] - prices_list[i-1]) 
                            for i in range(1, len(prices_list)))
        
        if total_movement == 0:
            return 1.0
            
        return net_change / total_movement
        
    def _calculate_volatility_regime(self, current_atr: float, price: float) -> float:
        """Normalize ATR to