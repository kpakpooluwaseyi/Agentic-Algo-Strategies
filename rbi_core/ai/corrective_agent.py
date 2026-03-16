"""rbi_core/ai/corrective_agent.py — RL parameter mutation with Sortino optimization."""
import copy
import numpy as np
from typing import Optional


class CorrectiveRLAgent:
    """
    Iteratively mutates strategy/execution parameters based on live PnL feedback.
    Reward function: continuous Sortino-based with proximity penalty near kill-switch.

    Key safety mechanisms:
    - EMA-smoothed gradients prevent oscillation.
    - Revert-to-best on consecutive negative rewards.
    - Hard clip bounds on all parameters.
    """

    PARAM_BOUNDS = {
        'atr_multiplier':            (0.5, 3.0),
        'max_holding_time_minutes':  (5, 480),
        'vol_threshold':             (100, 50000),
        'kill_switch_pnl_pct':       (-0.10, -0.01),
    }

    def __init__(self, learning_rate: float = 0.05, ema_alpha: float = 0.3,
                 max_consecutive_negatives: int = 3):
        self.learning_rate = learning_rate
        self.ema_alpha = ema_alpha  # Smoothing factor for gradient EMA
        self.max_consecutive_negatives = max_consecutive_negatives

        self.active_params: dict[str, float] = {
            'atr_multiplier': 1.5,
            'max_holding_time_minutes': 5,
            'vol_threshold': 1000,
            'kill_switch_pnl_pct': -0.05,
        }

        # Safety state
        self._best_params: dict[str, float] = copy.deepcopy(self.active_params)
        self._best_reward: float = float('-inf')
        self._consecutive_negatives: int = 0
        self._grad_ema: dict[str, float] = {k: 0.0 for k in self.active_params}
        self.swarm_degraded: bool = False  # Set True if Dell sync fails

    def step(self, current_sortino: float, current_pnl: float,
             raw_gradients: dict[str, float]) -> dict[str, float]:
        """
        Main RL step. Called after each trade batch or time interval.

        Args:
            current_sortino: Rolling Sortino ratio of recent trades.
            current_pnl: Current PnL as fraction of account (e.g., -0.03 = -3%).
            raw_gradients: Dict of param_name -> raw gradient estimate.

        Returns:
            Updated active_params dict.
        """
        reward = self._calculate_reward(current_sortino, current_pnl)

        # Track consecutive negatives
        if reward < 0:
            self._consecutive_negatives += 1
        else:
            self._consecutive_negatives = 0

        # Revert-to-best on sustained degradation
        if self._consecutive_negatives >= self.max_consecutive_negatives:
            self.active_params = copy.deepcopy(self._best_params)
            self._consecutive_negatives = 0
            self._grad_ema = {k: 0.0 for k in self.active_params}  # Reset EMA
            return copy.deepcopy(self.active_params)

        # Update best snapshot
        if reward > self._best_reward:
            self._best_reward = reward
            self._best_params = copy.deepcopy(self.active_params)

        # EMA-smooth gradients to prevent oscillation
        for param, raw_grad in raw_gradients.items():
            if param not in self._grad_ema:
                continue
            self._grad_ema[param] = (
                self.ema_alpha * raw_grad +
                (1 - self.ema_alpha) * self._grad_ema[param]
            )

        # Apply smoothed gradients scaled by reward magnitude
        reward_scale = np.clip(reward / 10.0, -1.0, 1.0)  # Normalize
        for param in self.active_params:
            if param in self._grad_ema:
                self.active_params[param] += (
                    self._grad_ema[param] * self.learning_rate * reward_scale
                )

        self._clip_parameters()
        return copy.deepcopy(self.active_params)

    def _calculate_reward(self, sortino: float, pnl: float) -> float:
        """
        Continuous reward function with kill-switch proximity penalty.
        No cliff — severity scales smoothly.
        """
        kill_pct = self.active_params['kill_switch_pnl_pct']

        # If breached: scale penalty by severity of breach
        if pnl < kill_pct:
            breach_severity = (kill_pct - pnl) / abs(kill_pct)
            return -100.0 * (1.0 + breach_severity)

        # Proximity penalty: penalize being CLOSE to kill-switch even if not breached
        if kill_pct < 0:
            # For negative kill thresholds (e.g., -0.05), pnl/kill_pct is:
            # 1.0 at threshold, 0.0 at breakeven, and >1 below threshold.
            proximity_ratio = max(0.0, min(1.0, pnl / kill_pct))
        else:
            proximity_ratio = 0.0
        if kill_pct != 0:
            proximity_penalty = proximity_ratio * 5.0
        else:
            proximity_penalty = 0.0

        return (sortino * 10.0) - proximity_penalty

    def _clip_parameters(self) -> None:
        for param, (lo, hi) in self.PARAM_BOUNDS.items():
            if param in self.active_params:
                self.active_params[param] = float(np.clip(self.active_params[param], lo, hi))

    def load_weights_from_swarm(self, swarm_params: dict[str, float]) -> None:
        """
        Apply externally computed parameter bounds from Nanobot WFA.
        Only updates bounds if swarm is not degraded.
        """
        if self.swarm_degraded:
            return
        # Merge: swarm provides updated bounds, we apply them
        for param, value in swarm_params.items():
            if param in self.PARAM_BOUNDS:
                lo, hi = self.PARAM_BOUNDS[param]
                self.active_params[param] = float(np.clip(value, lo, hi))
        self._best_params = copy.deepcopy(self.active_params)
        self._best_reward = float('-inf')  # Reset best after external update
