"""HeartbeatMonitor — 5s pulse / 15s timeout → SafeHarbor (Story 6.1)."""
from __future__ import annotations

import time


class HeartbeatMonitor:
    def __init__(self, timeout_seconds: int = 15) -> None:
        self._timeout = timeout_seconds
        self._last_beat: float = time.monotonic()

    def record_heartbeat(self) -> None:
        self._last_beat = time.monotonic()

    def is_alive(self) -> bool:
        return (time.monotonic() - self._last_beat) < self._timeout
