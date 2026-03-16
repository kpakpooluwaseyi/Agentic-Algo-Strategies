"""ATRCalculator — per-symbol Average True Range for Story 4.2."""
from __future__ import annotations

from typing import Optional


class ATRCalculator:
    @staticmethod
    def calculate(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        period: int = 14,
    ) -> Optional[float]:
        if len(closes) <= period:
            return None
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        if len(trs) < period:
            return None
        atr = sum(trs[-period:]) / period
        return float(atr)
