"""RiskGuardrails — enforces spread/leverage/drawdown limits for Story 4.3."""
from __future__ import annotations

from rbi_core.exceptions import RBIError


class RiskViolationError(RBIError):
    """A trade violated a configured risk guardrail."""


class RiskGuardrails:
    def __init__(
        self,
        max_spread_pct: float = 0.05,
        max_leverage: float = 10.0,
        max_drawdown_pct: float = 0.15,
    ) -> None:
        self.max_spread_pct = max_spread_pct
        self.max_leverage = max_leverage
        self.max_drawdown_pct = max_drawdown_pct

    def validate(
        self,
        spread_pct: float = 0.0,
        leverage: float = 1.0,
        current_drawdown_pct: float = 0.0,
    ) -> None:
        if spread_pct > self.max_spread_pct:
            raise RiskViolationError(
                f"spread {spread_pct:.4f} exceeds limit {self.max_spread_pct:.4f}"
            )
        if leverage > self.max_leverage:
            raise RiskViolationError(
                f"leverage {leverage} exceeds limit {self.max_leverage}"
            )
        if current_drawdown_pct > self.max_drawdown_pct:
            raise RiskViolationError(
                f"drawdown {current_drawdown_pct:.4f} exceeds limit {self.max_drawdown_pct:.4f}"
            )
