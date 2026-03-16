"""Unit tests for Epics 5 and 6 — Incubator Pool, Lifecycle, Resilience.

Stories covered:
  Epic 5:
    5.1 TickMultiplexer — fan-out ticks to N subscribers
    5.2 IncubatorFactory — spin up bot workers, cap at 50
    5.3 CullingEngine — cull bots below profit-factor threshold
  Epic 6:
    6.1 HeartbeatMonitor — 5s pulse, 15s timeout → SafeHarbor
    6.2 RegimeDetector — regime label (bull/bear/sideways/unknown)
    6.3 RegimeRotator — swap strategy pool on regime change
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from rbi_core.incubator.tick_multiplexer import TickMultiplexer
from rbi_core.incubator.incubator_factory import IncubatorFactory, BotWorker
from rbi_core.incubator.culling_engine import CullingEngine
from rbi_core.resilience.heartbeat_monitor import HeartbeatMonitor
from rbi_core.resilience.regime_detector import RegimeDetector, RegimeLabel
from rbi_core.resilience.regime_rotator import RegimeRotator


# ─── Epic 5.1 — TickMultiplexer ───────────────────────────────────────────────

class TestTickMultiplexer:

    def test_publish_calls_all_subscribers(self):
        mux = TickMultiplexer()
        cb1 = MagicMock()
        cb2 = MagicMock()
        mux.subscribe(cb1)
        mux.subscribe(cb2)
        tick = {"s": "BTC", "p": "40000"}
        mux.publish(tick)
        cb1.assert_called_once_with(tick)
        cb2.assert_called_once_with(tick)

    def test_subscriber_count(self):
        mux = TickMultiplexer()
        mux.subscribe(MagicMock())
        mux.subscribe(MagicMock())
        assert mux.subscriber_count == 2

    def test_unsubscribe_removes_callback(self):
        mux = TickMultiplexer()
        cb = MagicMock()
        mux.subscribe(cb)
        mux.unsubscribe(cb)
        mux.publish({"s": "ETH"})
        cb.assert_not_called()


# ─── Epic 5.2 — IncubatorFactory ──────────────────────────────────────────────

class TestIncubatorFactory:

    def test_spawn_creates_bot_worker(self):
        factory = IncubatorFactory(max_bots=50)
        bot = factory.spawn(dna={"fast": 8}, capital=1000.0)
        assert isinstance(bot, BotWorker)

    def test_cannot_exceed_max_bots(self):
        factory = IncubatorFactory(max_bots=2)
        factory.spawn({"a": 1}, 100.0)
        factory.spawn({"b": 2}, 100.0)
        with pytest.raises(RuntimeError, match="capacity"):
            factory.spawn({"c": 3}, 100.0)

    def test_bot_count_tracks_spawned(self):
        factory = IncubatorFactory(max_bots=10)
        factory.spawn({"a": 1}, 100.0)
        factory.spawn({"b": 2}, 100.0)
        assert factory.active_count == 2

    def test_kill_removes_bot(self):
        factory = IncubatorFactory(max_bots=5)
        bot = factory.spawn({"d": 4}, 100.0)
        factory.kill(bot.bot_id)
        assert factory.active_count == 0


# ─── Epic 5.3 — CullingEngine ─────────────────────────────────────────────────

class TestCullingEngine:

    def test_cull_below_threshold_returns_culled_ids(self):
        engine = CullingEngine(min_profit_factor=1.25, max_drawdown=0.15)
        bots = [
            {"bot_id": "a", "profit_factor": 0.9, "drawdown": 0.08},
            {"bot_id": "b", "profit_factor": 1.5, "drawdown": 0.05},
            {"bot_id": "c", "profit_factor": 1.1, "drawdown": 0.20},
        ]
        culled = engine.cull(bots)
        assert "a" in culled  # below profit_factor
        assert "b" not in culled  # passes
        assert "c" in culled  # exceeds drawdown

    def test_cull_returns_empty_when_all_pass(self):
        engine = CullingEngine(min_profit_factor=1.0, max_drawdown=0.5)
        bots = [{"bot_id": "x", "profit_factor": 2.0, "drawdown": 0.01}]
        assert engine.cull(bots) == []


# ─── Epic 6.1 — HeartbeatMonitor ──────────────────────────────────────────────

class TestHeartbeatMonitor:

    def test_is_alive_true_within_timeout(self):
        monitor = HeartbeatMonitor(timeout_seconds=15)
        monitor.record_heartbeat()
        assert monitor.is_alive() is True

    def test_is_alive_false_after_timeout(self):
        import time
        monitor = HeartbeatMonitor(timeout_seconds=0)
        time.sleep(0.01)
        assert monitor.is_alive() is False


# ─── Epic 6.2 — RegimeDetector ────────────────────────────────────────────────

class TestRegimeDetector:

    def test_returns_bull_on_rising_trend(self):
        closes = [100 + i for i in range(30)]
        label = RegimeDetector.detect(closes)
        assert label == RegimeLabel.BULL

    def test_returns_bear_on_falling_trend(self):
        closes = [200 - i for i in range(30)]
        label = RegimeDetector.detect(closes)
        assert label == RegimeLabel.BEAR

    def test_returns_sideways_on_flat_data(self):
        closes = [100.0] * 30
        label = RegimeDetector.detect(closes)
        assert label == RegimeLabel.SIDEWAYS

    def test_returns_unknown_on_insufficient_data(self):
        label = RegimeDetector.detect([100.0, 101.0])
        assert label == RegimeLabel.UNKNOWN


# ─── Epic 6.3 — RegimeRotator ─────────────────────────────────────────────────

class TestRegimeRotator:

    def test_rotate_swaps_strategy_pool(self):
        pools = {
            RegimeLabel.BULL: ["trend_follow"],
            RegimeLabel.BEAR: ["short_sell"],
        }
        rotator = RegimeRotator(strategy_pools=pools)
        strategies = rotator.get_pool_for(RegimeLabel.BULL)
        assert strategies == ["trend_follow"]

    def test_rotate_returns_empty_list_for_unknown_regime(self):
        rotator = RegimeRotator(strategy_pools={})
        assert rotator.get_pool_for(RegimeLabel.UNKNOWN) == []
