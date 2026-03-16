"""RegimeDetector — simple trend-based regime classifier (Story 6.2)."""
from __future__ import annotations

from enum import Enum


class RegimeLabel(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class RegimeDetector:
    @staticmethod
    def detect(closes: list[float], min_periods: int = 20) -> RegimeLabel:
        if len(closes) < min_periods:
            return RegimeLabel.UNKNOWN
        half = len(closes) // 2
        first_half_mean = sum(closes[:half]) / half
        second_half_mean = sum(closes[half:]) / (len(closes) - half)
        delta = (second_half_mean - first_half_mean) / (first_half_mean or 1)
        if delta > 0.02:
            return RegimeLabel.BULL
        if delta < -0.02:
            return RegimeLabel.BEAR
        return RegimeLabel.SIDEWAYS
