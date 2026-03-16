"""BatchSimulator — sequential VectorBT backtest runner for Mac M1.

Runs backtests one ticker at a time (no parallelism) and calls
``gc.collect()`` after each to keep peak RAM below 6 GB.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbi_core.exceptions import SimulationError

logger = logging.getLogger(__name__)


@dataclass
class SimResult:
    """Minimal backtest result for a single ticker."""

    ticker: str
    total_return: float
    sharpe: float
    max_drawdown: float


class BatchSimulator:
    """Sequential VectorBT batch simulation with explicit GC."""

    def __init__(self, memory_limit_gb: float = 6.0) -> None:
        self._memory_limit_bytes = memory_limit_gb * 1024 ** 3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy_dna: dict[str, Any],
        parquet_dir: str,
        tickers: list[str],
    ) -> dict[str, SimResult]:
        """Run backtests sequentially across *tickers*.

        Args:
            strategy_dna: Indicator params and thresholds.
            parquet_dir:  Root directory containing ``{TICKER}/1m.parquet`` files.
            tickers:      Ordered list of ticker symbols to simulate.

        Returns:
            Mapping of ticker → `SimResult` for each successful simulation.
            Failed tickers are logged and excluded (fail-forward).
        """
        results: dict[str, SimResult] = {}

        for ticker in tickers:
            parquet_path = Path(parquet_dir) / ticker / "1m.parquet"
            try:
                result = self._simulate_ticker(strategy_dna, parquet_path)
                results[ticker] = result
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "simulation_ticker_failed",
                    extra={
                        "event": "simulation_ticker_failed",
                        "ticker": ticker,
                        "error": str(exc),
                    },
                )
            finally:
                gc.collect()

        return results

    # ------------------------------------------------------------------
    # Internal — can be mocked in tests; real VectorBT call goes here
    # ------------------------------------------------------------------

    def _simulate_ticker(
        self,
        strategy_dna: dict[str, Any],
        parquet_path: Path,
    ) -> SimResult:
        """Run the VectorBT backtest for a single ticker.

        Override or mock this method in tests.  Production usage requires
        VectorBT installed on the Mac Validator environment.

        Args:
            strategy_dna: Strategy parameters.
            parquet_path: Path to the 1-minute Parquet file.

        Returns:
            A populated `SimResult`.

        Raises:
            SimulationError: If VectorBT raises or Parquet is unreadable.
        """
        try:
            import pandas as pd
            import vectorbt as vbt  # noqa: F401 — Mac-side dependency

            df = pd.read_parquet(str(parquet_path))
            # Minimal: run a moving-average crossover as a placeholder.
            fast = strategy_dna.get("fast", 8)
            slow = strategy_dna.get("slow", 21)
            fast_ma = df["close"].rolling(fast).mean()
            slow_ma = df["close"].rolling(slow).mean()
            entries = fast_ma > slow_ma
            exits = fast_ma <= slow_ma

            pf = vbt.Portfolio.from_signals(df["close"], entries, exits)
            return SimResult(
                ticker=parquet_path.parent.name,
                total_return=float(pf.total_return()),
                sharpe=float(pf.sharpe_ratio()),
                max_drawdown=float(pf.max_drawdown()),
            )
        except ImportError:
            raise SimulationError(
                "VectorBT is not installed. "
                "Install it on the Mac Validator: pip install vectorbt"
            )
        except Exception as exc:
            raise SimulationError(
                f"Simulation failed for {parquet_path}: {exc}"
            ) from exc
