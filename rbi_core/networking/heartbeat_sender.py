"""HeartbeatSender — runs on Mac Validator.

Sends a heartbeat message to `stream:heartbeat` every HEARTBEAT_INTERVAL_S seconds.
Errors in publish do NOT crash the sender — they are logged as warnings and retried
on the next cycle.

Architecture refs:
- NFR8: Heartbeat 5s cadence
- stream:heartbeat (architecture § Naming Patterns)
- source field: "mac_validator"
"""
from __future__ import annotations

import asyncio

from rbi_core.networking.redis_streams import RedisStreamsClient
from rbi_core.utils.logging import get_logger

logger = get_logger("rbi_core.networking.heartbeat_sender")

HEARTBEAT_INTERVAL_S: int = 5


class HeartbeatSender:
    """Sends periodic heartbeat events to Redis Streams.

    Designed to run as a background asyncio Task on the Mac Validator.

    Usage:
        sender = HeartbeatSender(streams_client=client, source="mac_validator")
        task = asyncio.create_task(sender.start())
        ...
        await sender.stop()
        task.cancel()
    """

    def __init__(
        self,
        streams_client: RedisStreamsClient,
        source: str = "mac_validator",
        interval_s: int = HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._client = streams_client
        self._source = source
        self._interval = interval_s
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Begin heartbeat loop. Runs until `stop()` is called."""
        self._stop_event.clear()
        
        while not self._stop_event.is_set():
            try:
                await self._client.publish(
                    "heartbeat",
                    {"alive": True},
                    source=self._source,
                )
                logger.debug(
                    "heartbeat_sent",
                    extra={"event": "heartbeat_sent", "context": {"source": self._source}},
                )
            except Exception as exc:
                logger.warning(
                    "heartbeat_publish_failed",
                    extra={"event": "heartbeat_failed", "context": {"error": str(exc)}},
                )
                
            try:
                # Use event.wait with a timeout to allow instantaneous shutdown interruption
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass  # normal interval elapsed, continue loop

    async def stop(self) -> None:
        """Signal the heartbeat loop to stop immediately."""
        self._stop_event.set()
