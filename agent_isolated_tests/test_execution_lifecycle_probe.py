import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rbi_core.execution.risk import PortfolioExecutionQueue
from rbi_core.execution.trade_manager import TradeManager


def test_exec_slots_reopen_after_trade_close_dashboard_flow():
    """
    Expected behavior: after trades are closed, queue slots should be released so
    new signals can be accepted.

    This reproduces the patched dashboard flow, where closed TradeManager trades
    release PortfolioExecutionQueue slots via trade_id -> position_key mapping.
    """
    exec_queue = PortfolioExecutionQueue(initial_equity=100.0, max_concurrent=2)
    trade_to_position = {}

    def on_trade_closed(result: dict):
        position_key = trade_to_position.pop(result["trade_id"], None)
        if position_key:
            exec_queue.close_position(position_key, result["realized_pnl"])
        else:
            exec_queue.update_equity(exec_queue.equity + result["realized_pnl"])

    tm = TradeManager(on_trade_closed=on_trade_closed)

    params = {"atr_multiplier": 1.0, "max_holding_time_minutes": 1.0}
    market = {"price": 100.0, "atr": 1.0}

    # Fill both queue slots
    p1 = exec_queue.submit_signal({"action": "BUY", "strategy": "A"}, params, market)
    assert p1
    t1 = tm.open_trade("BUY", 100.0, 1.0, "A", 1.0, 1.0, max_holding_minutes=1.0)
    trade_to_position[t1] = p1

    p2 = exec_queue.submit_signal({"action": "BUY", "strategy": "B"}, params, market)
    assert p2
    t2 = tm.open_trade("BUY", 100.0, 1.0, "B", 1.0, 1.0, max_holding_minutes=1.0)
    trade_to_position[t2] = p2

    # Trades close in TradeManager lifecycle
    tm.close_trade(t1, exit_price=101.0, reason="manual")
    tm.close_trade(t2, exit_price=101.0, reason="manual")

    # Desired behavior: should now accept another signal because two trades closed.
    assert exec_queue.submit_signal({"action": "BUY", "strategy": "C"}, params, market)
