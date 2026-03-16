"""RedisEventPublisher — publishes strategy_ready events to Redis Streams.

Publishes ONLY the passport URI, never raw DNA (architecture constraint).
"""
from __future__ import annotations

import logging

from rbi_core.passport.passport_compiler import StrategyPassport

logger = logging.getLogger(__name__)

_STREAM = "stream:strategy_ready"


class RedisEventPublisher:
    """Publishes Strategy Passport URI events to Redis Streams."""

    def __init__(self, streams_client) -> None:
        self._client = streams_client

    async def publish_strategy_ready(self, passport: StrategyPassport) -> bytes:
        """Publish a ``strategy_ready`` event containing ONLY the passport URI.

        Args:
            passport: The compiled ``StrategyPassport``.

        Returns:
            The Redis message ID returned by XADD.
        """
        payload = {
            "uri": passport.uri,
            "ticker": passport.ticker,
            "passport_id": passport.passport_id,
        }
        msg_id = await self._client.publish("strategy_ready", payload)
        logger.info(
            "strategy_ready_published",
            extra={
                "event": "strategy_ready_published",
                "passport_id": passport.passport_id,
                "ticker": passport.ticker,
                "uri": passport.uri,
            },
        )
        return msg_id
