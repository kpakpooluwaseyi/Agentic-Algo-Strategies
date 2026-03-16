"""RegimeRotator — maps regime labels to strategy pools (Story 6.3)."""
from __future__ import annotations

from rbi_core.resilience.regime_detector import RegimeLabel


class RegimeRotator:
    def __init__(self, strategy_pools: dict) -> None:
        self._pools = strategy_pools

    def get_pool_for(self, regime: RegimeLabel) -> list:
        return self._pools.get(regime, [])
