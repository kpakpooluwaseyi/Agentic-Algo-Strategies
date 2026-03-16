"""rbi_core/strategy/base.py — Abstract base class for all strategies."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Signal:
    """Standardized signal output from any strategy."""
    action: str          # "BUY", "SELL", or "HOLD"
    confidence: float    # 0.0 to 1.0
    meta: dict = field(default_factory=dict)  # strategy-specific metadata


class BaseStrategy(ABC):
    """
    Contract for all strategies in the pool.
    Every strategy MUST subclass this and implement on_tick().
    """

    def __init__(self, name: str):
        self.name: str = name
        self.current_confidence: float = 0.0
        self._enabled: bool = True

    @abstractmethod
    def on_tick(self, tick_data: dict) -> Optional[Signal]:
        """
        Process a single tick of market data.

        Args:
            tick_data: dict with fields (use .get() for optional fields):
                Core (always present):
                - 'price': float
                - 'volume': float
                - 'timestamp': float (unix epoch)
                - 'atr': float (current ATR value)
                - 'bid': float
                - 'ask': float
                Enriched (may be absent on older data feeds):
                - 'session': str ('asia'|'london'|'newyork'|'overlap_london_ny')
                - 'spread': float (ask - bid)
                - 'spread_zscore': float (spread vs rolling mean)
                - 'vwap': float (volume-weighted average price)
                - 'tick_imbalance': float (buy-sell pressure)
                - 'regime': str ('trending'|'ranging'|'volatile'|'unknown')
                - 'atr_percentile': float (0-100, volatility rank)
                - 'daily_bar_count': int (ticks since session open)
                - 'position_count': int (currently open positions)
                - 'daily_pnl': float (realized PnL today)

        Returns:
            Signal if the strategy has an actionable opinion, else None.
        """
        ...


    @abstractmethod
    def reset(self) -> None:
        """Reset internal state. Called when RL agent prunes this strategy."""
        ...

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    @property
    def is_enabled(self) -> bool:
        return self._enabled
