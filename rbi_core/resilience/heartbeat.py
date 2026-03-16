"""HeartbeatMonitor — runs on Dell Executor.

Monitors `stream:heartbeat` via XREAD. If no heartbeat is received for
MISSED_HEARTBEAT_THRESHOLD × HEARTBEAT_INTERVAL_S seconds, raises HeartbeatError
to trigger Safe Harbor Mode (NFR8, FR17).

Architecture refs:
- NFR8: 3 consecutive 5s failures → Safe Harbor
- HeartbeatError in rbi_core.exceptions
- stream:heartbeat (architecture § Naming Patterns)
"""
from __future__ import annotations

import redis.asyncio

from rbi_core.exceptions import HeartbeatError
from rbi_core.utils.logging import get_logger

logger = get_logger("rbi_core.resilience.heartbeat")

MISSED_HEARTBEAT_THRESHOLD: int = 3
_BLOCK_MS: int = 5500  # slightly > 5s so we don't miss a heartbeat on edge timing


class HeartbeatMonitor:
    """Monitors `stream:heartbeat` and raises HeartbeatError after 3 consecutive misses.

    Designed to run as a background asyncio Task on the Dell Executor.

    The miss counter is reset to 0 any time a valid heartbeat arrives. Only
    CONSECUTIVE misses count toward the threshold (NFR8).

    Raises:
        HeartbeatError: After MISSED_HEARTBEAT_THRESHOLD consecutive empty reads.
    """

    def __init__(self, redis_client: redis.asyncio.Redis) -> None:
        self._redis = redis_client
        self._miss_count: int = 0

    def reset_counter(self) -> None:
        """Reset the consecutive miss counter. Called on valid heartbeat receipt."""
        self._miss_count = 0

    async def run(self) -> None:
        """Block on stream:heartbeat. Raises HeartbeatError on 3 consecutive misses."""
        while True:
            results = await self._redis.xread(
                {"stream:heartbeat": "$"},
                count=1,
                block=_BLOCK_MS,
            )
            if results:
                self.reset_counter()
                logger.debug(
                    "heartbeat_received",
                    extra={"event": "heartbeat_received", "context": {}},
                )
            else:
                self._miss_count += 1
                logger.warning(
                    "heartbeat_missed",
                    extra={
                        "event": "heartbeat_missed",
                        "context": {
                            "miss_count": self._miss_count,
                            "threshold": MISSED_HEARTBEAT_THRESHOLD,
                        },
                    },
                )
                if self._miss_count >= MISSED_HEARTBEAT_THRESHOLD:
                    logger.error(
                        "heartbeat_lost",
                        extra={
                            "event": "heartbeat_lost",
                            "context": {"consecutive_misses": self._miss_count},
                        },
                    )
                    raise HeartbeatError(
                        f"Heartbeat lost after {self._miss_count} consecutive missed intervals"
                    )
