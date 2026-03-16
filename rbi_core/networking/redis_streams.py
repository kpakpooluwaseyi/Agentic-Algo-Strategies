"""Redis Streams client for Mac↔Dell async messaging.

Architecture refs:
- Stream names: stream:{event_type} → stream:strategy_ready, stream:heartbeat
- Consumer groups: dell_executor, mac_validator
- Payload format: MessagePack binary, carried in `payload` field
- MAXLEN ~ 1000 (approximate trim to avoid performance hits)
- Message fields: type, payload (msgpack bytes), ts (unix str), checksum (sha256[:8])
"""
from __future__ import annotations

import hashlib
import time
from typing import AsyncIterator, Optional

import msgpack
import redis.asyncio
import redis.exceptions

from rbi_core.exceptions import StreamConsumeError, StreamPublishError
from rbi_core.utils.logging import get_logger

logger = get_logger("rbi_core.networking.redis_streams")

# Stream keys (from architecture § Naming Patterns)
STREAM_STRATEGY_READY = "stream:strategy_ready"
STREAM_HEARTBEAT = "stream:heartbeat"

# Consumer groups (from architecture § Communication Patterns)
GROUP_DELL_EXECUTOR = "dell_executor"
GROUP_MAC_VALIDATOR = "mac_validator"

# Stream size cap
STREAM_MAXLEN = 1000


class RedisStreamsClient:
    """Async pub/sub client over Redis Streams 8.0.

    Wraps XADD / XREADGROUP / XACK via redis.asyncio.Redis.
    All payloads are MessagePack-encoded binary blobs.
    """

    def __init__(self, redis_client: redis.asyncio.Redis) -> None:
        self._redis = redis_client

    async def initialize_streams(self) -> None:
        """Create all required streams and consumer groups idempotently.

        Safe to call multiple times — BUSYGROUP errors are silently swallowed.
        Called on service startup before any publish or consume operations.
        """
        for stream, group in [
            (STREAM_STRATEGY_READY, GROUP_DELL_EXECUTOR),
            (STREAM_HEARTBEAT, GROUP_DELL_EXECUTOR),
            (STREAM_STRATEGY_READY, GROUP_MAC_VALIDATOR),
            (STREAM_HEARTBEAT, GROUP_MAC_VALIDATOR),
        ]:
            await self._create_group(stream, group)

        logger.info(
            "streams_initialized",
            extra={"event": "streams_initialized", "context": {"streams": [STREAM_STRATEGY_READY, STREAM_HEARTBEAT]}},
        )

    async def _create_group(self, stream: str, group: str) -> None:
        """XGROUP CREATE with MKSTREAM. Silently ignores BUSYGROUP."""
        try:
            await self._redis.xgroup_create(stream, group, id="$", mkstream=True)
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                return  # Already exists — idempotent, OK
            raise

    async def publish(self, event_type: str, data: dict, source: Optional[str] = None) -> bytes:
        """XADD a MessagePack-encoded message to stream:{event_type}.

        Args:
            event_type: Logical event name (e.g., "heartbeat", "strategy_ready")
            data:       Python dict to be MessagePack-encoded as the payload.
            source:     Optional component source name (e.g., "mac_validator")

        Returns:
            Redis message ID (bytes), e.g. b"1740000000000-0"
        """
        try:
            payload = msgpack.packb(data, use_bin_type=True)
            checksum = hashlib.sha256(payload).hexdigest()[:8]
            
            fields = {
                "type": event_type,
                "payload": payload,
                "ts": str(int(time.time() * 1000)),
                "checksum": checksum,
            }
            if source:
                fields["source"] = source
                
            msg_id: bytes = await self._redis.xadd(
                f"stream:{event_type}",
                fields,
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
            return msg_id
        except redis.exceptions.RedisError as exc:
            raise StreamPublishError(f"Failed to publish to stream:{event_type}: {exc}") from exc

    async def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 5000,
    ) -> AsyncIterator[tuple[bytes, dict]]:
        """Async generator: XREADGROUP loop yielding (msg_id, fields) tuples.
        First consumes pending messages (PEL) never ACKed by this consumer,
        then switches to consuming new messages.

        Args:
            stream:    Full stream name (e.g., "stream:heartbeat")
            group:     Consumer group name
            consumer:  Unique consumer identity (e.g., "worker-1")
            count:     Max messages per XREADGROUP call
            block_ms:  Milliseconds to block waiting for messages
        """
        try:
            # Step 1: Recover pending messages for this consumer (starts at "0")
            last_id = "0"
            while True:
                messages = await self._redis.xreadgroup(
                    group,
                    consumer,
                    {stream: last_id},
                    count=count,
                    block=10,  # return immediately if no pending
                )
                if not messages:
                    break  # No more pending messages

                entries = messages[0][1]
                if not entries:
                    break

                for msg_id, fields in entries:
                    yield msg_id, fields

            # Step 2: Normal consumption of new messages (starts at ">")
            while True:
                messages = await self._redis.xreadgroup(
                    group,
                    consumer,
                    {stream: ">"},
                    count=count,
                    block=block_ms,
                )
                if not messages:
                    continue
                for _stream_name, entries in messages:
                    for msg_id, fields in entries:
                        yield msg_id, fields
        except redis.exceptions.RedisError as exc:
            raise StreamConsumeError(f"Consume error on {stream}: {exc}") from exc

    async def ack(self, stream: str, group: str, msg_id: bytes) -> None:
        """XACK a message, marking it as delivered to the consumer group."""
        await self._redis.xack(stream, group, msg_id)
        logger.debug(
            "message_acked",
            extra={"event": "message_acked", "context": {"stream": stream, "msg_id": str(msg_id)}},
        )
