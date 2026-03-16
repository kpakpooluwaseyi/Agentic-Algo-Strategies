"""rbi_core/strategy/pool/ema_scalp.py — Example EMA Scalp strategy."""
from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque


class EMAScalp(BaseStrategy):
    """Simple EMA crossover scalper. Placeholder for real logic."""

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        super().__init__(name="EMA_Scalp")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._prices: deque = deque(maxlen=slow_period + 1)

    def on_tick(self, tick_data: dict) -> Optional[Signal]:
        self._prices.append(tick_data['price'])
        if len(self._prices) < self.slow_period:
            return None
        # Placeholder: real EMA math goes here
        self.current_confidence = 0.0
        return None

    def reset(self) -> None:
        self._prices.clear()
        self.current_confidence = 0.0
