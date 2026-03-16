"""Unit tests for BatchSimulator — TDD RED phase.

VectorBT is not installed in the test venv; all _simulate_ticker
calls are mocked via unittest.mock.patch.object.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from rbi_core.simulation.batch_simulator import BatchSimulator, SimResult
from rbi_core.exceptions import SimulationError


DNA = {"indicator": "EMA", "fast": 8, "slow": 21, "threshold": 0.01}


@pytest.fixture
def mock_sim_result():
    return SimResult(ticker="BTC", total_return=0.25, sharpe=1.8, max_drawdown=0.12)


class TestSimResult:

    def test_has_required_fields(self):
        r = SimResult(ticker="ETH", total_return=0.1, sharpe=1.0, max_drawdown=0.05)
        assert r.ticker == "ETH"
        assert r.total_return == 0.1
        assert r.sharpe == 1.0
        assert r.max_drawdown == 0.05


class TestBatchSimulatorRun:

    def test_returns_result_for_each_ticker(self, tmp_path, mock_sim_result):
        sim = BatchSimulator()
        with patch.object(BatchSimulator, "_simulate_ticker", return_value=mock_sim_result):
            results = sim.run(DNA, parquet_dir=str(tmp_path), tickers=["BTC", "ETH"])
        assert set(results.keys()) == {"BTC", "ETH"}

    def test_all_results_are_sim_result_instances(self, tmp_path, mock_sim_result):
        sim = BatchSimulator()
        with patch.object(BatchSimulator, "_simulate_ticker", return_value=mock_sim_result):
            results = sim.run(DNA, parquet_dir=str(tmp_path), tickers=["BTC"])
        assert isinstance(results["BTC"], SimResult)

    def test_calls_simulate_ticker_for_each_asset(self, tmp_path, mock_sim_result):
        sim = BatchSimulator()
        with patch.object(BatchSimulator, "_simulate_ticker", return_value=mock_sim_result) as mock_sim:
            sim.run(DNA, parquet_dir=str(tmp_path), tickers=["BTC", "ETH", "SOL"])
        assert mock_sim.call_count == 3

    def test_gc_collect_called_after_each_ticker(self, tmp_path, mock_sim_result):
        sim = BatchSimulator()
        with patch.object(BatchSimulator, "_simulate_ticker", return_value=mock_sim_result):
            with patch("gc.collect") as mock_gc:
                sim.run(DNA, parquet_dir=str(tmp_path), tickers=["BTC", "ETH"])
        # gc.collect must be called at least once per ticker
        assert mock_gc.call_count >= 2

    def test_fail_forward_on_single_ticker_error(self, tmp_path, mock_sim_result):
        sim = BatchSimulator()

        def side_effect(dna, parquet_path):
            if "ETH" in str(parquet_path):
                raise RuntimeError("VectorBT crash")
            return mock_sim_result

        with patch.object(BatchSimulator, "_simulate_ticker", side_effect=side_effect):
            results = sim.run(DNA, parquet_dir=str(tmp_path), tickers=["BTC", "ETH", "SOL"])
        # ETH failed — should NOT be in results; BTC and SOL should be
        assert "BTC" in results
        assert "SOL" in results
        assert "ETH" not in results

    def test_empty_ticker_list_returns_empty_dict(self, tmp_path):
        sim = BatchSimulator()
        results = sim.run(DNA, parquet_dir=str(tmp_path), tickers=[])
        assert results == {}

    def test_processes_tickers_sequentially_not_parallel(self, tmp_path, mock_sim_result):
        """Verify no threading: call order must match the ticker list order."""
        call_order = []

        def capture_call(dna, parquet_path):
            ticker = parquet_path.parent.name
            call_order.append(ticker)
            return mock_sim_result

        sim = BatchSimulator()
        tickers = ["BTC", "ETH", "SOL"]
        with patch.object(BatchSimulator, "_simulate_ticker", side_effect=capture_call):
            sim.run(DNA, parquet_dir=str(tmp_path), tickers=tickers)
        assert call_order == tickers
