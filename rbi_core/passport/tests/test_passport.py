"""Unit tests for PassportCompiler and RedisEventPublisher — TDD RED phase."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rbi_core.passport.passport_compiler import PassportCompiler, StrategyPassport
from rbi_core.passport.event_publisher import RedisEventPublisher


SAMPLE_DNA = {"fast": 8, "slow": 21, "threshold": 0.01}


class TestStrategyPassport:

    def test_has_required_fields(self):
        p = StrategyPassport(
            passport_id="test-123",
            dna=SAMPLE_DNA,
            ticker="BTC",
            total_return=0.3,
            sharpe=1.5,
            max_drawdown=0.1,
            uri="/golden/test-123.json",
            created_at=datetime.now(timezone.utc),
        )
        assert p.passport_id == "test-123"
        assert p.dna == SAMPLE_DNA
        assert p.uri == "/golden/test-123.json"


class TestPassportCompilerMint:

    def test_mint_returns_passport(self, tmp_path):
        compiler = PassportCompiler(golden_dir=str(tmp_path))
        passport = compiler.mint(
            dna=SAMPLE_DNA,
            ticker="BTC",
            total_return=0.3,
            sharpe=1.5,
            max_drawdown=0.1,
        )
        assert isinstance(passport, StrategyPassport)

    def test_mint_creates_file_in_golden_dir(self, tmp_path):
        compiler = PassportCompiler(golden_dir=str(tmp_path))
        passport = compiler.mint(SAMPLE_DNA, "ETH", 0.2, 1.2, 0.08)
        assert (tmp_path / f"{passport.passport_id}.json").exists()

    def test_passport_id_is_unique_per_call(self, tmp_path):
        compiler = PassportCompiler(golden_dir=str(tmp_path))
        p1 = compiler.mint(SAMPLE_DNA, "BTC", 0.1, 1.0, 0.05)
        p2 = compiler.mint(SAMPLE_DNA, "BTC", 0.1, 1.0, 0.05)
        assert p1.passport_id != p2.passport_id

    def test_uri_references_golden_dir(self, tmp_path):
        compiler = PassportCompiler(golden_dir=str(tmp_path))
        passport = compiler.mint(SAMPLE_DNA, "SOL", 0.15, 0.9, 0.07)
        assert str(tmp_path) in passport.uri


class TestRedisEventPublisher:

    @pytest.mark.asyncio
    async def test_publishes_strategy_ready_event(self):
        mock_client = MagicMock()
        mock_client.publish = AsyncMock(return_value=b"1-0")
        publisher = RedisEventPublisher(streams_client=mock_client)
        passport = StrategyPassport(
            passport_id="abc",
            dna=SAMPLE_DNA,
            ticker="BTC",
            total_return=0.3,
            sharpe=1.5,
            max_drawdown=0.1,
            uri="/golden/abc.json",
            created_at=datetime.now(timezone.utc),
        )
        await publisher.publish_strategy_ready(passport)
        mock_client.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_payload_contains_only_uri_not_raw_data(self):
        mock_client = MagicMock()
        mock_client.publish = AsyncMock(return_value=b"1-0")
        publisher = RedisEventPublisher(streams_client=mock_client)
        passport = StrategyPassport(
            passport_id="xyz",
            dna=SAMPLE_DNA,
            ticker="ETH",
            total_return=0.1,
            sharpe=0.8,
            max_drawdown=0.04,
            uri="/golden/xyz.json",
            created_at=datetime.now(timezone.utc),
        )
        await publisher.publish_strategy_ready(passport)
        call_args = mock_client.publish.call_args
        payload = call_args[0][1]  # second positional arg = data dict
        # URI must be in the payload
        assert "uri" in payload
        # Raw strategy DNA MUST NOT be in the stream payload
        assert "dna" not in payload
        assert "fast" not in str(payload)
