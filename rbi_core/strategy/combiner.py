"""rbi_core/strategy/combiner.py — Multi-strategy concurrent evaluator with conflict resolution."""
import os
import concurrent.futures
from typing import Optional
from rbi_core.strategy.base import BaseStrategy, Signal


class CombinedSignal:
    """Aggregated, conflict-resolved output from the combiner."""
    def __init__(self, action: str, net_confidence: float, contributing_strategies: list[str]):
        self.action = action                           # "BUY", "SELL", or "HOLD"
        self.net_confidence = net_confidence
        self.contributing_strategies = contributing_strategies


class StrategyCombiner:
    """
    Runs enabled strategies in parallel on each tick.
    Resolves BUY/SELL conflicts via confidence-weighted voting.
    """

    DIRECTION_MAP = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}

    def __init__(
        self,
        strategies: list[BaseStrategy],
        consensus_threshold: float = 0.3,
        regime_weight: float = 1.0,
    ):
        """
        Args:
            strategies: Instantiated strategy objects (must subclass BaseStrategy).
            consensus_threshold: Minimum |net_score| to produce a non-HOLD signal.
            regime_weight: Multiplier from PicoClaw regime score (0.0 to 2.0).
                           Updated externally via set_regime_weight().
        """
        self.strategies = strategies
        self.consensus_threshold = consensus_threshold
        self.regime_weight = regime_weight
        max_workers = min(len(strategies), os.cpu_count() or 4)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def set_regime_weight(self, weight: float) -> None:
        """Called by picoclaw_ingest when a new regime score arrives."""
        self.regime_weight = max(0.0, min(2.0, weight))  # Clamp [0, 2]

    def evaluate_tick(self, tick_data: dict) -> Optional[CombinedSignal]:
        """
        Push tick to all enabled strategies in parallel.
        Returns CombinedSignal if consensus is reached, else None.
        """
        enabled = [s for s in self.strategies if s.is_enabled]
        if not enabled:
            return None

        # Dispatch all strategies
        future_to_strat = {
            self._pool.submit(s.on_tick, tick_data): s
            for s in enabled
        }

        signals: list[tuple[str, float, str]] = []  # (action, confidence, strategy_name)
        for future in concurrent.futures.as_completed(future_to_strat):
            strat = future_to_strat[future]
            try:
                result: Optional[Signal] = future.result(timeout=2.0)
                if result and result.action != "HOLD":
                    signals.append((result.action, result.confidence, strat.name))
            except Exception as e:
                print(f"[Combiner] Error in {strat.name}: {e}")

        if not signals:
            return None

        # Confidence-weighted voting
        net_score = 0.0
        for action, confidence, _ in signals:
            direction = self.DIRECTION_MAP.get(action, 0.0)
            net_score += direction * confidence * self.regime_weight

        if abs(net_score) < self.consensus_threshold:
            return None  # No consensus — HOLD

        final_action = "BUY" if net_score > 0 else "SELL"
        contributors = [name for action, _, name in signals if self.DIRECTION_MAP.get(action, 0) * (1 if final_action == "BUY" else -1) > 0]

        return CombinedSignal(
            action=final_action,
            net_confidence=abs(net_score),
            contributing_strategies=contributors,
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
