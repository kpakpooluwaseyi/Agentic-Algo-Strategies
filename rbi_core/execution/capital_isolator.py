"""CapitalIsolator — per-strategy capital and leverage guardrails (Story 4.5)."""
from __future__ import annotations

from rbi_core.exceptions import RBIError


class CapitalViolationError(RBIError):
    """Order exceeds strategy's capital or leverage allocation."""


class CapitalIsolator:
    def __init__(self, allocated_capital: float, max_leverage: float = 5.0) -> None:
        self._capital = allocated_capital
        self._max_leverage = max_leverage

    def validate_order(self, notional: float, leverage: float = 1.0) -> None:
        if notional > self._capital:
            raise CapitalViolationError(
                f"Order notional {notional} exceeds allocated capital {self._capital}"
            )
        if leverage > self._max_leverage:
            raise CapitalViolationError(
                f"Requested leverage {leverage} exceeds max {self._max_leverage}"
            )
