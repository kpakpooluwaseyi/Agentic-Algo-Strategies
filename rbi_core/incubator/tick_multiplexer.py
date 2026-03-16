"""TickMultiplexer — fan-out single WebSocket feed to N subscribers (Story 5.1)."""
from __future__ import annotations

from typing import Callable


class TickMultiplexer:
    def __init__(self) -> None:
        self._subscribers: list[Callable] = []

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self, callback: Callable) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        self._subscribers = [c for c in self._subscribers if c is not callback]

    def publish(self, tick: dict) -> None:
        for cb in self._subscribers:
            cb(tick)
