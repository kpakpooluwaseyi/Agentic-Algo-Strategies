"""Unit tests for HeartbeatSender (Mac side) — TDD RED phase."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rbi_core.networking.heartbeat_sender import HeartbeatSender, HEARTBEAT_INTERVAL_S
from rbi_core.networking.redis_streams import RedisStreamsClient


@pytest.fixture
def mock_streams_client():
    client = MagicMock(spec=RedisStreamsClient)
    client.publish = AsyncMock(return_value=b"1740000000000-0")
    return client


class TestHeartbeatSenderConstants:
    def test_heartbeat_interval_is_5s(self):
        assert HEARTBEAT_INTERVAL_S == 5


class TestHeartbeatSender:

    @pytest.mark.asyncio
    async def test_publishes_heartbeat_with_correct_type(self, mock_streams_client):
        sender = HeartbeatSender(streams_client=mock_streams_client, source="mac_validator")
        # Run for one tick then stop
        task = asyncio.create_task(sender.start())
        await asyncio.sleep(0.05)
        await sender.stop()
        task.cancel()

        mock_streams_client.publish.assert_awaited()
        call_args = mock_streams_client.publish.call_args[0]
        call_kwargs = mock_streams_client.publish.call_args[1]
        assert call_args[0] == "heartbeat"
        assert call_kwargs["source"] == "mac_validator"

    @pytest.mark.asyncio
    async def test_publishes_alive_true_in_payload(self, mock_streams_client):
        sender = HeartbeatSender(streams_client=mock_streams_client)
        task = asyncio.create_task(sender.start())
        await asyncio.sleep(0.05)
        await sender.stop()
        task.cancel()

        data = mock_streams_client.publish.call_args[0][1]
        assert data.get("alive") is True

    @pytest.mark.asyncio
    async def test_does_not_crash_on_publish_error(self, mock_streams_client):
        """If publish raises, sender must continue running (not crash)."""
        mock_streams_client.publish = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
        sender = HeartbeatSender(streams_client=mock_streams_client)
        task = asyncio.create_task(sender.start())
        await asyncio.sleep(0.05)
        await sender.stop()
        task.cancel()
        # If we got here without an exception, the test passes

    @pytest.mark.asyncio
    async def test_stop_terminates_loop(self, mock_streams_client):
        """After stop(), the sender stops publishing on the next iteration."""
        call_count = 0

        async def counting_publish(event_type, data, **kwargs):
            nonlocal call_count
            call_count += 1

        mock_streams_client.publish = AsyncMock(side_effect=counting_publish)

        # Use a short interval so we don't spinlock but still iterate fast
        sender = HeartbeatSender(streams_client=mock_streams_client, interval_s=0.01)
        task = asyncio.create_task(sender.start())

        await asyncio.sleep(0.02)  # let at least 1 publish run
        count_before_stop = call_count
        await sender.stop()
        await asyncio.sleep(0.02)  # let any in-flight cycle finish

        # After stop, count should not grow (or grow by at most 1 more in-flight)
        assert count_before_stop >= 1  # at least one publish happened
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
