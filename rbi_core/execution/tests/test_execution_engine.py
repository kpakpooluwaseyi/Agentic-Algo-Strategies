"""Unit tests for Epic 4 — Execution Engine components.

Covers:
  - TickNormalizer (Story 4.1)
  - ATRCalculator (Story 4.2)
  - RiskGuardrails (Story 4.3)
  - PriorityQueue (Story 4.4)
  - CapitalIsolator (Story 4.5)
  - WALStateManager (Story 4.6)
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from rbi_core.execution.tick_normalizer import TickNormalizer, NormalizedTick
from rbi_core.execution.atr_calculator import ATRCalculator
from rbi_core.execution.risk_guardrails import RiskGuardrails, RiskViolationError
from rbi_core.execution.priority_queue import TradePriorityQueue, TradeRequest, TradeType
from rbi_core.execution.capital_isolator import CapitalIsolator, CapitalViolationError
from rbi_core.execution.wal_state_manager import WALStateManager


# ─── Story 4.1 — TickNormalizer ───────────────────────────────────────────────

class TestTickNormalizer:

    def test_normalize_returns_normalized_tick(self):
        raw = {"s": "BTCUSDT", "p": "42000.5", "q": "0.01", "T": 1700000000000}
        tick = TickNormalizer.normalize(raw)
        assert isinstance(tick, NormalizedTick)

    def test_normalized_tick_has_required_fields(self):
        raw = {"s": "BTCUSDT", "p": "42000.5", "q": "0.01", "T": 1700000000000}
        tick = TickNormalizer.normalize(raw)
        assert tick.symbol == "BTCUSDT"
        assert tick.price == 42000.5
        assert tick.quantity == 0.01

    def test_normalize_raises_on_missing_price(self):
        with pytest.raises((KeyError, ValueError)):
            TickNormalizer.normalize({"s": "BTC", "q": "1"})


# ─── Story 4.2 — ATRCalculator ────────────────────────────────────────────────

class TestATRCalculator:

    def test_calculate_returns_float(self):
        closes = [100.0 + i * 0.5 for i in range(20)]
        highs  = [c + 1 for c in closes]
        lows   = [c - 1 for c in closes]
        atr = ATRCalculator.calculate(highs, lows, closes, period=14)
        assert isinstance(atr, float)
        assert atr > 0

    def test_returns_none_when_insufficient_data(self):
        result = ATRCalculator.calculate([100], [99], [100], period=14)
        assert result is None

    def test_higher_volatility_yields_higher_atr(self):
        closes = [100.0] * 20
        tight_atr = ATRCalculator.calculate(
            [c + 0.1 for c in closes], [c - 0.1 for c in closes], closes, period=14
        )
        wide_atr = ATRCalculator.calculate(
            [c + 5.0 for c in closes], [c - 5.0 for c in closes], closes, period=14
        )
        assert wide_atr > tight_atr


# ─── Story 4.3 — RiskGuardrails ───────────────────────────────────────────────

class TestRiskGuardrails:

    def test_passes_when_all_within_limits(self):
        rg = RiskGuardrails(max_spread_pct=0.05, max_leverage=10.0, max_drawdown_pct=0.15)
        rg.validate(spread_pct=0.01, leverage=5.0, current_drawdown_pct=0.05)

    def test_raises_on_spread_violation(self):
        rg = RiskGuardrails(max_spread_pct=0.02)
        with pytest.raises(RiskViolationError, match="spread"):
            rg.validate(spread_pct=0.05)

    def test_raises_on_leverage_violation(self):
        rg = RiskGuardrails(max_leverage=5.0)
        with pytest.raises(RiskViolationError, match="leverage"):
            rg.validate(leverage=10.0)

    def test_raises_on_drawdown_violation(self):
        rg = RiskGuardrails(max_drawdown_pct=0.10)
        with pytest.raises(RiskViolationError, match="drawdown"):
            rg.validate(current_drawdown_pct=0.20)


# ─── Story 4.4 — PriorityQueue ────────────────────────────────────────────────

class TestTradePriorityQueue:

    def test_close_requests_dequeued_before_open(self):
        q = TradePriorityQueue()
        q.push(TradeRequest(symbol="BTC", trade_type=TradeType.OPEN, size=1.0))
        q.push(TradeRequest(symbol="BTC", trade_type=TradeType.CLOSE, size=1.0))
        first = q.pop()
        assert first.trade_type == TradeType.CLOSE

    def test_pop_returns_none_when_empty(self):
        q = TradePriorityQueue()
        assert q.pop() is None

    def test_push_and_pop_multiple(self):
        q = TradePriorityQueue()
        q.push(TradeRequest(symbol="ETH", trade_type=TradeType.OPEN, size=0.5))
        q.push(TradeRequest(symbol="SOL", trade_type=TradeType.CLOSE, size=2.0))
        q.push(TradeRequest(symbol="BNB", trade_type=TradeType.OPEN, size=1.0))
        first = q.pop()
        assert first.trade_type == TradeType.CLOSE


# ─── Story 4.5 — CapitalIsolator ──────────────────────────────────────────────

class TestCapitalIsolator:

    def test_allows_order_within_allocation(self):
        isolator = CapitalIsolator(allocated_capital=10_000.0, max_leverage=5.0)
        isolator.validate_order(notional=5_000.0, leverage=2.0)  # must not raise

    def test_raises_capital_violation_on_oversize(self):
        isolator = CapitalIsolator(allocated_capital=1_000.0, max_leverage=5.0)
        with pytest.raises(CapitalViolationError, match="capital"):
            isolator.validate_order(notional=2_000.0, leverage=1.0)

    def test_raises_capital_violation_on_overleveraged(self):
        isolator = CapitalIsolator(allocated_capital=10_000.0, max_leverage=5.0)
        with pytest.raises(CapitalViolationError, match="leverage"):
            isolator.validate_order(notional=5_000.0, leverage=10.0)


# ─── Story 4.6 — WALStateManager ──────────────────────────────────────────────

class TestWALStateManager:

    def test_save_and_load_state(self, tmp_path):
        db_path = str(tmp_path / "state.db")
        manager = WALStateManager(db_path=db_path)
        manager.save_position(symbol="BTC", size=1.0, entry_price=40000.0)
        positions = manager.load_positions()
        assert any(p["symbol"] == "BTC" for p in positions)

    def test_load_positions_returns_list(self, tmp_path):
        db_path = str(tmp_path / "state.db")
        manager = WALStateManager(db_path=db_path)
        result = manager.load_positions()
        assert isinstance(result, list)

    def test_position_overwritten_on_second_save(self, tmp_path):
        db_path = str(tmp_path / "state.db")
        manager = WALStateManager(db_path=db_path)
        manager.save_position("ETH", 1.0, 2000.0)
        manager.save_position("ETH", 2.0, 2100.0)  # update
        positions = manager.load_positions()
        eth = next(p for p in positions if p["symbol"] == "ETH")
        assert eth["size"] == 2.0
