"""Unit tests for HeartbeatMonitor (Dell side) — TDD RED phase."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from rbi_core.resilience.heartbeat import HeartbeatMonitor, MISSED_HEARTBEAT_THRESHOLD
from rbi_core.exceptions import HeartbeatError


@pytest.fixture
def mock_redis():
    r = MagicMock()
    r.xread = AsyncMock(return_value=[])
    return r


class TestHeartbeatMonitorConstants:
    def test_missed_threshold_is_3(self):
        assert MISSED_HEARTBEAT_THRESHOLD == 3


class TestHeartbeatMonitor:

    @pytest.mark.asyncio
    async def test_raises_heartbeat_error_after_3_misses(self, mock_redis):
        """Three consecutive empty xread results → HeartbeatError."""
        mock_redis.xread = AsyncMock(return_value=[])
        monitor = HeartbeatMonitor(redis_client=mock_redis)
        with pytest.raises(HeartbeatError, match="Heartbeat lost"):
            await monitor.run()

    @pytest.mark.asyncio
    async def test_resets_miss_count_on_valid_heartbeat(self, mock_redis):
        """A valid heartbeat resets the counter so 3 misses don't include earlier non-consecutive ones."""
        import msgpack
        payload = msgpack.packb({"alive": True}, use_bin_type=True)
        valid_entry = [(b"stream:heartbeat", [(b"1740000-0", {b"type": b"heartbeat", b"payload": payload})])]

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return valid_entry   # First call: heartbeat received, miss_count resets
            elif call_count < 5:
                return []            # Next 3 calls: no heartbeat
            else:
                raise HeartbeatError("forced stop")

        mock_redis.xread = AsyncMock(side_effect=side_effect)
        monitor = HeartbeatMonitor(redis_client=mock_redis)

        with pytest.raises(HeartbeatError):
            await monitor.run()

        # The monitor should have seen 4 calls: 1 good + 3 misses → raise
        assert call_count >= 4

    @pytest.mark.asyncio
    async def test_does_not_raise_before_3_consecutive_misses(self, mock_redis):
        """2 misses should not raise. Only after 3 consecutive misses."""
        import msgpack
        payload = msgpack.packb({"alive": True}, use_bin_type=True)
        valid_entry = [(b"stream:heartbeat", [(b"1740000-0", {b"type": b"heartbeat", b"payload": payload})])]

        call_count = 0
        raises_called = False

        async def side_effect(*args, **kwargs):
            nonlocal call_count, raises_called
            call_count += 1
            if call_count <= 2:
                return []        # 2 misses
            elif call_count == 3:
                return valid_entry  # reset on 3rd call
            else:
                raises_called = True
                raise HeartbeatError("forced termination")

        mock_redis.xread = AsyncMock(side_effect=side_effect)
        monitor = HeartbeatMonitor(redis_client=mock_redis)

        # Should NOT raise HeartbeatError before 3 consecutive misses
        with pytest.raises(HeartbeatError, match="forced termination"):
            await monitor.run()

        assert raises_called is True  # we reached call 4 without premature raise
