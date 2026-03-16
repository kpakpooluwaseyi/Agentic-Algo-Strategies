"""Unit tests for MonteCarloEngine — TDD RED phase."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rbi_core.analysis.monte_carlo import MonteCarloEngine, MonteCarloStats


@pytest.fixture
def base_returns():
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.001, 0.02, size=100))


class TestMonteCarloEngineRun:

    def test_output_shape(self, base_returns):
        result = MonteCarloEngine.run(base_returns, n=200, spread_bps=5.0)
        assert result.shape == (200, len(base_returns))

    def test_output_is_ndarray(self, base_returns):
        result = MonteCarloEngine.run(base_returns, n=50, spread_bps=0.0)
        assert isinstance(result, np.ndarray)

    def test_spread_bps_zero_does_not_alter_mean(self, base_returns):
        result_no_spread = MonteCarloEngine.run(base_returns, n=500, spread_bps=0.0)
        # With no spread, mean should be close to input mean
        assert abs(result_no_spread.mean() - base_returns.mean()) < 0.005

    def test_spread_bps_reduces_returns(self, base_returns):
        result_no_spread = MonteCarloEngine.run(base_returns, n=500, spread_bps=0.0)
        result_with_spread = MonteCarloEngine.run(base_returns, n=500, spread_bps=10.0)
        # Higher spread should lower mean return
        assert result_with_spread.mean() < result_no_spread.mean()

    def test_n_zero_returns_empty_array(self, base_returns):
        result = MonteCarloEngine.run(base_returns, n=0, spread_bps=5.0)
        assert result.shape[0] == 0


class TestMonteCarloStats:

    def test_from_simulation_mean_total_return(self, base_returns):
        sim = MonteCarloEngine.run(base_returns, n=1000, spread_bps=0.0)
        stats = MonteCarloStats.from_simulation(sim)
        assert isinstance(stats.mean_total_return, float)

    def test_worst_case_is_below_mean(self, base_returns):
        sim = MonteCarloEngine.run(base_returns, n=1000, spread_bps=0.0)
        stats = MonteCarloStats.from_simulation(sim)
        assert stats.worst_case_return <= stats.mean_total_return

    def test_best_case_is_above_mean(self, base_returns):
        sim = MonteCarloEngine.run(base_returns, n=1000, spread_bps=0.0)
        stats = MonteCarloStats.from_simulation(sim)
        assert stats.best_case_return >= stats.mean_total_return

    def test_stats_has_required_fields(self, base_returns):
        sim = MonteCarloEngine.run(base_returns, n=100, spread_bps=2.0)
        stats = MonteCarloStats.from_simulation(sim)
        assert hasattr(stats, "mean_total_return")
        assert hasattr(stats, "worst_case_return")
        assert hasattr(stats, "best_case_return")
