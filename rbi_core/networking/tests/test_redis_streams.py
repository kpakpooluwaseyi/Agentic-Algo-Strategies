"""Unit tests for rbi_core.networking.redis_streams — TDD RED phase."""
from __future__ import annotations

import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Imports will fail until redis_streams.py is implemented (RED)
from rbi_core.networking.redis_streams import RedisStreamsClient
from rbi_core.exceptions import StreamPublishError, StreamConsumeError


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    """Async mock of redis.asyncio.Redis."""
    r = MagicMock()
    r.xadd = AsyncMock(return_value=b"1740000000000-0")
    r.xgroup_create = AsyncMock(return_value=True)
    r.xreadgroup = AsyncMock(return_value=[])
    r.xack = AsyncMock(return_value=1)
    r.xread = AsyncMock(return_value=[])
    return r


@pytest.fixture
def client(mock_redis):
    return RedisStreamsClient(redis_client=mock_redis)


# ─── Tests: Stream Initialization ────────────────────────────────────────────

class TestInitializeStreams:

    @pytest.mark.asyncio
    async def test_creates_strategy_ready_group(self, client, mock_redis):
        await client.initialize_streams()
        mock_redis.xgroup_create.assert_any_await(
            "stream:strategy_ready", "dell_executor", id="$", mkstream=True
        )

    @pytest.mark.asyncio
    async def test_creates_heartbeat_group(self, client, mock_redis):
        await client.initialize_streams()
        mock_redis.xgroup_create.assert_any_await(
            "stream:heartbeat", "dell_executor", id="$", mkstream=True
        )

    @pytest.mark.asyncio
    async def test_idempotent_on_busygroup_error(self, client, mock_redis):
        """BUSYGROUP error must be silently swallowed — not re-raised."""
        import redis.exceptions
        mock_redis.xgroup_create = AsyncMock(
            side_effect=redis.exceptions.ResponseError("BUSYGROUP Consumer Group name already exists")
        )
        # Should NOT raise
        await client.initialize_streams()

    @pytest.mark.asyncio
    async def test_reraises_non_busygroup_error(self, client, mock_redis):
        """Other ResponseErrors must propagate."""
        import redis.exceptions
        mock_redis.xgroup_create = AsyncMock(
            side_effect=redis.exceptions.ResponseError("WRONGTYPE")
        )
        with pytest.raises(redis.exceptions.ResponseError, match="WRONGTYPE"):
            await client.initialize_streams()


# ─── Tests: Publish ──────────────────────────────────────────────────────────

class TestPublish:

    @pytest.mark.asyncio
    async def test_returns_message_id(self, client, mock_redis):
        msg_id = await client.publish("heartbeat", {"alive": True})
        assert msg_id == b"1740000000000-0"

    @pytest.mark.asyncio
    async def test_xadd_called_with_correct_stream(self, client, mock_redis):
        await client.publish("strategy_ready", {"uri": "/data/strat.json"})
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "stream:strategy_ready"

    @pytest.mark.asyncio
    async def test_xadd_fields_contain_type(self, client, mock_redis):
        await client.publish("heartbeat", {"alive": True})
        fields = mock_redis.xadd.call_args[0][1]
        assert fields["type"] == "heartbeat"

    @pytest.mark.asyncio
    async def test_xadd_fields_contain_ts(self, client, mock_redis):
        before_ms = int(time.time() * 1000) - 100
        await client.publish("heartbeat", {"alive": True})
        fields = mock_redis.xadd.call_args[0][1]
        assert int(fields["ts"]) >= before_ms
        assert len(fields["ts"]) >= 13  # Ms precision length

    @pytest.mark.asyncio
    async def test_xadd_fields_contain_msgpack_payload(self, client, mock_redis):
        import msgpack
        await client.publish("heartbeat", {"alive": True})
        fields = mock_redis.xadd.call_args[0][1]
        decoded = msgpack.unpackb(fields["payload"], raw=False)
        assert decoded["alive"] is True

    @pytest.mark.asyncio
    async def test_xadd_fields_contain_checksum(self, client, mock_redis):
        await client.publish("heartbeat", {"alive": True})
        fields = mock_redis.xadd.call_args[0][1]
        assert len(fields["checksum"]) == 8

    @pytest.mark.asyncio
    async def test_xadd_uses_maxlen_1000(self, client, mock_redis):
        await client.publish("heartbeat", {"alive": True})
        kwargs = mock_redis.xadd.call_args[1]
        assert kwargs.get("maxlen") == 1000

    @pytest.mark.asyncio
    async def test_xadd_uses_approximate_trim(self, client, mock_redis):
        await client.publish("heartbeat", {"alive": True})
        kwargs = mock_redis.xadd.call_args[1]
        assert kwargs.get("approximate") is True


    @pytest.mark.asyncio
    async def test_xadd_includes_source_if_provided(self, client, mock_redis):
        await client.publish("heartbeat", {"alive": True}, source="mac_validator")
        fields = mock_redis.xadd.call_args[0][1]
        assert fields["source"] == "mac_validator"


# ─── Tests: Consume ──────────────────────────────────────────────────────────

class TestConsume:

    @pytest.mark.asyncio
    async def test_yields_msg_id_and_fields(self, client, mock_redis):
        import msgpack
        payload = msgpack.packb({"alive": True}, use_bin_type=True)
        
        # consume() queries "0" first for pending, we return empty there,
        # then queries ">" for new, where we return the message.
        async def side_effect(*args, **kwargs):
            streams = args[2] if len(args) > 2 else kwargs.get("streams", {})
            if list(streams.values())[0] == "0":
                return []
            return [
                (b"stream:heartbeat", [(b"1740000000000-0", {b"type": b"heartbeat", b"payload": payload})])
            ]
            
        mock_redis.xreadgroup = AsyncMock(side_effect=side_effect)
        results = []
        async for msg_id, fields in client.consume("stream:heartbeat", "dell_executor", "worker-1"):
            results.append((msg_id, fields))
            break  # only consume one
        assert results[0][0] == b"1740000000000-0"
        assert b"payload" in results[0][1]

    @pytest.mark.asyncio
    async def test_empty_result_does_not_yield(self, client, mock_redis):
        """Empty XREADGROUP results should yield nothing; loop should keep running."""
        results = []
        call_count = 0

        class _StopTest(Exception):
            """Sentinel to break out of the infinite consume loop in tests."""

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            
            # Allow the "0" read to return empty
            streams = args[2] if len(args) > 2 else kwargs.get("streams", {})
            if list(streams.values())[0] == "0":
                return []
                
            # For the ">" reads
            call_count += 1
            if call_count >= 2:
                raise _StopTest
            return []
        mock_redis.xreadgroup = AsyncMock(side_effect=side_effect)
        try:
            async for _ in client.consume("stream:heartbeat", "dell_executor", "worker-1"):
                results.append(1)
        except _StopTest:
            pass
        assert len(results) == 0


# ─── Tests: ACK ──────────────────────────────────────────────────────────────

class TestAck:

    @pytest.mark.asyncio
    async def test_xack_called_with_correct_args(self, client, mock_redis):
        await client.ack("stream:strategy_ready", "dell_executor", b"1740000000000-0")
        mock_redis.xack.assert_awaited_once_with(
            "stream:strategy_ready", "dell_executor", b"1740000000000-0"
        )


# ─── Tests: redis.conf configuration ─────────────────────────────────────────

class TestRedisConfFile:

    def test_redis_conf_has_maxmemory_4gb(self):
        with open("config/redis.conf") as f:
            content = f.read()
        assert "maxmemory 4gb" in content

    def test_redis_conf_has_appendfsync_everysec(self):
        with open("config/redis.conf") as f:
            content = f.read()
        assert "appendfsync everysec" in content

    def test_redis_conf_has_aof_enabled(self):
        with open("config/redis.conf") as f:
            content = f.read()
        assert "appendonly yes" in content
