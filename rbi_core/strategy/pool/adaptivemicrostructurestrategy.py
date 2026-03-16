from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Any, List
from rbi_core.strategy.base import BaseStrategy, Signal

@dataclass
class MicrostructureState:
    """Internal state container for strategy memory"""
    prices: deque
    volumes: deque
    atrs: deque
    spreads: deque
    tick_count: int
    consecutive_buys: int
    consecutive_sells: int
    last_micro_price: float

class AdaptiveMicrostructureStrategy(BaseStrategy):
    """
    Novel strategy combining volume-weighted microstructure analysis,
    ATR-normalized adaptive momentum, and order flow pressure indicators.
    """
    
    def __init__(self, lookback: int = 20, volume_lookback: int = 10,
                 atr_threshold: float = 1.5, spread_filter: float = 0.002,
                 momentum_weight: float = 0.35, pressure_weight: float = 0.35,
                 micro_weight: float = 0.30):
        super().__init__()
        self.lookback = lookback
        self.volume_lookback = volume_lookback
        self.atr_threshold = atr_threshold
        self.spread_filter = spread_filter
        self.momentum_weight = momentum_weight
        self.pressure_weight = pressure_weight
        self.micro_weight = micro_weight
        
        self.state = MicrostructureState(
            prices=deque(maxlen=lookback),
            volumes=deque(maxlen=volume_lookback),
            atrs=deque(maxlen=lookback),
            spreads=deque(maxlen=lookback),
            tick_count=0,
            consecutive_buys=0,
            consecutive_sells=0,
            last_micro_price=0.0
        )
        
    def reset(self):
        """Reset all stateful buffers and counters to initial state"""
        self.state.prices.clear()
        self.state.volumes.clear()
        self.state.atrs.clear()
        self.state.spreads.clear()
        self.state.tick_count = 0
        self.state.consecutive_buys = 0
        self.state.consecutive_sells = 0
        self.state.last_micro_price = 0.0
        
    def _calculate_volume_weighted_micro_price(self, bid: float, ask: float, 
                                                volume: float, prev_volume: float) -> float:
        """Calculate micro-price adjusted by volume delta and spread positioning"""
        if bid <= 0 or ask <= 0 or ask <= bid:
            return (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
            
        spread = ask - bid
        mid = (bid + ask) / 2
        
        if prev_volume <= 0:
            return mid
            
        volume_delta = (volume - prev_volume) / prev_volume
        volume_delta = max(-1.0, min(1.0, volume_delta))
        
        # Weight by position within spread based on volume pressure
        micro_adjustment = spread * 0.5 * volume_delta
        micro_price = mid + micro_adjustment
        
        return micro_price
        
    def _calculate_adaptive_momentum_score(self, price: float, atr: float) -> float:
        """Calculate ATR-normalized momentum with volatility regime adjustment"""
        if len(self.state.prices) < 2 or atr <= 0:
            return 0.0
            
        # Price change normalized by ATR (Keltner-style)
        price_change = price - self.state.prices[-1]
        normalized_change = price_change / atr
        
        # Historical volatility context
        if len(self.state.atrs) >= self.lookback // 2:
            atr_list = list(self.state.atrs)
            avg_atr = sum(atr_list) / len(atr_list)
            
            if avg_atr > 0:
                # Regime detection: high vs low volatility
                regime_ratio = atr / avg_atr
                if regime_ratio > self.atr_threshold:
                    # High volatility regime - dampen momentum
                    normalized_change *= 0.6
                    
                # Z-score calculation for momentum extremity
                if len(self.state.prices) >= self.lookback:
                    returns = []
                    prices_list = list(self.state.prices)
                    for i in range(1, len(prices_list)):
                        if prices_list[i-1] != 0:
                            returns.append((prices_list[i] - prices_list[i-1]) / prices_list[i-1])
                    
                    if returns:
                        avg_ret = sum(returns) / len(returns)
                        variance = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
                        std_ret = sqrt(variance) if variance > 0 else 0.001
                        
                        current