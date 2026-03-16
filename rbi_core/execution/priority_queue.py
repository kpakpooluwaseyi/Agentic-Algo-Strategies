"""TradePriorityQueue — CLOSE requests dequeued before OPEN (Story 4.4)."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class TradeType(IntEnum):
    CLOSE = 0  # lower enum value = higher priority
    OPEN = 1


@dataclass(order=True)
class TradeRequest:
    trade_type: TradeType = field(default=TradeType.OPEN)
    symbol: str = field(default="", compare=False)
    size: float = field(default=0.0, compare=False)


class TradePriorityQueue:
    def __init__(self) -> None:
        self._heap: list[TradeRequest] = []

    def push(self, request: TradeRequest) -> None:
        heapq.heappush(self._heap, request)

    def pop(self) -> Optional[TradeRequest]:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)
