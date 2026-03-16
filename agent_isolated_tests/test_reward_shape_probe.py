import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rbi_core.ai.corrective_agent import CorrectiveRLAgent


def test_reward_should_improve_as_pnl_moves_farther_from_kill_switch():
    """
    Expected behavior: with equal Sortino, a safer PnL (farther from kill-switch)
    should have a higher reward than a near-threshold PnL.
    """
    agent = CorrectiveRLAgent()

    near_kill = agent._calculate_reward(sortino=0.0, pnl=-0.049)  # kill is -0.05
    far_from_kill = agent._calculate_reward(sortino=0.0, pnl=0.0)

    assert far_from_kill > near_kill
