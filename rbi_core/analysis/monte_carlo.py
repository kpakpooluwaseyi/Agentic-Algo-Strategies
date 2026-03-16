"""MonteCarloEngine — dynamic slippage Monte Carlo simulation.

Generates Gaussian-noise synthetic return paths using the input
return series as a template, then deducts spread costs in basis points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloStats:
    """Aggregated statistics over a Monte Carlo simulation array."""

    mean_total_return: float
    worst_case_return: float   # 5th percentile of total path returns
    best_case_return: float    # 95th percentile of total path returns

    @classmethod
    def from_simulation(cls, sim_array: np.ndarray) -> "MonteCarloStats":
        """Compute stats from a simulation array of shape (n, timesteps).

        Total return for each path = sum of per-period returns.
        """
        total_returns = sim_array.sum(axis=1)
        return cls(
            mean_total_return=float(np.mean(total_returns)),
            worst_case_return=float(np.percentile(total_returns, 5)),
            best_case_return=float(np.percentile(total_returns, 95)),
        )


class MonteCarloEngine:
    """Produces synthetic return path arrays via Gaussian noise injection."""

    @staticmethod
    def run(
        returns: pd.Series,
        n: int,
        spread_bps: float = 0.0,
    ) -> np.ndarray:
        """Run *n* Monte Carlo simulations from *returns*.

        Args:
            returns:    Base 1-period return series (e.g. from Parquet OHLCV).
            n:          Number of simulation paths.
            spread_bps: Spread cost in basis points deducted from each return.

        Returns:
            Array of shape ``(n, len(returns))``.  Each row is one path.
        """
        if n == 0:
            return np.empty((0, len(returns)))

        base = returns.to_numpy()
        std = float(np.std(base)) or 1e-6

        rng = np.random.default_rng()
        noise = rng.normal(loc=0.0, scale=std, size=(n, len(base)))
        sim = base + noise              # broadcast: each row gets the same base
        sim -= spread_bps / 10_000     # spread deduction element-wise

        return sim
