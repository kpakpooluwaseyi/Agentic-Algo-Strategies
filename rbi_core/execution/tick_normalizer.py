"""TickNormalizer — normalizes raw Binance WebSocket tick data."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NormalizedTick:
    symbol: str
    price: float
    quantity: float
    timestamp_ms: int


class TickNormalizer:
    @staticmethod
    def normalize(raw: dict) -> NormalizedTick:
        return NormalizedTick(
            symbol=str(raw["s"]),
            price=float(raw["p"]),
            quantity=float(raw["q"]),
            timestamp_ms=int(raw["T"]),
        )
