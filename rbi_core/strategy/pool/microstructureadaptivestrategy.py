from rbi_core.strategy.base import BaseStrategy, Signal
from collections import deque
from typing import Dict, Any, Literal
import math


class MicrostructureAdaptiveStrategy(BaseStrategy):
    """
    Hybrid regime-switching strategy combining market microstructure analysis 
    with adaptive volatility-normalized momentum.
    """
    
    def __init__(self, lookback: int = 20, vwap_window: int = 50, 
                 momentum_window: int = 5, spread_filter_pct: float = 0.3):
        super().__init__()
        self.lookback = lookback
        self.vwap_window = vwap_window
        self.momentum_window = momentum_window
        self.spread_filter_pct = spread_filter_pct
        
        # Price and volatility history
        self.price_history = deque(maxlen=lookback)
        self.atr_history = deque(maxlen=lookback)
        self.spread_history = deque(maxlen=lookback)
        
        # VWAP calculation state
        self.vwap_prices = deque(maxlen=vwap_window)
        self.vwap_volumes = deque(maxlen=vwap_window)
        
        # Momentum and return tracking
        self.returns = deque(maxlen=momentum_window)
        self.atr_normalized_returns = deque(maxlen=lookback)
        
        # Position and signal state
        self.last_price: float = 0.0
        self.last_timestamp: float = 0.0
        self.current_position: Literal[-1, 0, 1] = 0
        self.consecutive_same_signals: int = 0
        self.volatility_regime: Literal['low', 'medium', 'high'] = 'medium'
        
    def reset(self) -> None:
        """Clear all stateful buffers and counters."""
        self.price_history.clear()
        self.atr_history.clear()
        self.spread_history.clear()
        self.vwap_prices.clear()
        self.vwap_volumes.clear()
        self.returns.clear()
        self.atr_normalized_returns.clear()
        self.last_price = 0.0
        self.last_timestamp = 0.0
        self.current_position = 0
        self.consecutive_same_signals = 0
        self.volatility_regime = 'medium'
        
    def on_tick(self, tick_data: Dict[str, Any]) -> Signal:
        price = float(tick_data['price'])
        volume = float(tick_data['volume'])
        timestamp = float(tick_data