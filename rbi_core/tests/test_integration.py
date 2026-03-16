"""rbi_core/tests/test_integration.py — End-to-end integration test with live Hyperliquid data."""
import os
import sys
import time
import tempfile
import pytest

# Ensure rbi_core importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rbi_core.data.collectors.buffer_mgr import TickBuffer
from rbi_core.data.collectors.hyperliquid_ws import HyperliquidWSFeed
from rbi_core.strategy.combiner import StrategyCombiner
from rbi_core.strategy.pool.ema_scalp import EMAScalp
from rbi_core.strategy.pool.rsi_divergence import RSIDivergence
from rbi_core.strategy.pool.vwap_mean_reversion import VWAPMeanReversion
from rbi_core.strategy.pool.bollinger_scalp import BollingerScalp
from rbi_core.ai.corrective_agent import CorrectiveRLAgent
from rbi_core.execution.risk import PortfolioExecutionQueue
from rbi_core.execution.dispatcher import OrderDispatcher
from rbi_core.swarm_integration.nanoclaw_monitor import NanoClawMonitor


def run_integration_test(duration_s: int = 20):
    """Run live data through the full pipeline for `duration_s` seconds."""
    print("=" * 60)
    print("  INTEGRATION TEST — Live Tick Pipeline")
    print("=" * 60)

    # --- Setup ---
    db_path = os.path.join(tempfile.gettempdir(), "rbi_test_ticks.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    tick_buffer = TickBuffer(db_path)
    tick_buffer.start()

    # Shared state for NanoClaw
    state = {
        'last_tick_ts': time.time(),
        'spread': 0.01,
        'pnl': 0.0,
        'ticks_received': 0,
        'signals_generated': 0,
        'orders_submitted': 0,
    }

    def on_tick_wrapper(tick: dict):
        """Route ticks to buffer and update integration state."""
        tick_buffer.append_tick(tick)
        state['last_tick_ts'] = time.time()
        state['ticks_received'] += 1

        bid = tick.get('bid', 0)
        ask = tick.get('ask', 0)
        if bid and ask:
            state['spread'] = ask - bid

        # Feed to combiner
        combined = combiner.evaluate_tick(tick)
        if combined:
            state['signals_generated'] += 1
            # Pass to risk queue
            signal_dict = {
                'action': combined.action,
                'strategy': ','.join(combined.contributing_strategies),
            }
            ok = exec_queue.submit_signal(
                signal_dict,
                rl_agent.active_params,
                {'price': tick['price'], 'atr': max(tick.get('atr', 1.0), 0.01)},
            )
            if ok:
                state['orders_submitted'] += 1

    # WebSocket
    ws_feed = HyperliquidWSFeed(symbols=['BTC'], on_tick=on_tick_wrapper)

    # Strategies
    strategies = [
        EMAScalp(),
        RSIDivergence(bar_ticks=20),  # Lower bar_ticks for faster signal in test
        VWAPMeanReversion(min_ticks_for_vwap=50, entry_deviation_pct=0.1),
        BollingerScalp(bar_ticks=15, period=10),
    ]
    combiner = StrategyCombiner(strategies, consensus_threshold=0.1)

    # RL Agent
    rl_agent = CorrectiveRLAgent()

    # Execution
    exec_queue = PortfolioExecutionQueue(initial_equity=100.0)
    dispatcher = OrderDispatcher(exec_queue, mode="mock")
    dispatcher.start()

    # NanoClaw
    nanoclaw = NanoClawMonitor(
        halt_callback=exec_queue.halt,
        get_last_tick_ts=lambda: state['last_tick_ts'],
        get_current_spread=lambda: state['spread'],
        get_current_pnl=lambda: state['pnl'],
        tick_stale_threshold_s=30.0,  # Lenient for test
    )
    nanoclaw.start()

    # --- Run ---
    print(f"\nStarting live test for {duration_s}s...")
    ws_feed.start()
    time.sleep(duration_s)

    # --- Shutdown ---
    ws_feed.stop()
    nanoclaw.stop()
    dispatcher.stop()
    tick_buffer.stop()
    combiner.shutdown()
    time.sleep(1)

    # --- Report ---
    print("\n" + "=" * 60)
    print("  INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"  Ticks received:     {state['ticks_received']}")
    print(f"  Signals generated:  {state['signals_generated']}")
    print(f"  Orders submitted:   {state['orders_submitted']}")
    print(f"  NanoClaw halted:    {exec_queue.halted}")
    print(f"  RL params:          atr_mult={rl_agent.active_params['atr_multiplier']:.3f}")

    # Verify DB persistence
    stored = tick_buffer.query_recent('BTC', limit=10)
    print(f"  DB stored (last 10): {len(stored)} rows")

    # Assertions
    assert state['ticks_received'] > 0, "No ticks received"
    assert len(stored) > 0, "No ticks persisted to DB"
    assert not exec_queue.halted, "NanoClaw halted unexpectedly"
    print("\n  ✅ ALL ASSERTIONS PASSED")
    print("=" * 60)

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)
        wal = db_path + "-wal"
        shm = db_path + "-shm"
        for f in [wal, shm]:
            if os.path.exists(f):
                os.remove(f)


@pytest.mark.integration
def test_integration_live_smoke():
    if os.environ.get("RBI_RUN_LIVE_INTEGRATION") != "1":
        pytest.skip("Set RBI_RUN_LIVE_INTEGRATION=1 to run live websocket test")
    duration_s = int(os.environ.get("RBI_INTEGRATION_DURATION_S", "20"))
    run_integration_test(duration_s=duration_s)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGALRM, lambda s, f: sys.exit(1))
    signal.alarm(45)  # Hard kill after 45s
    run_integration_test(duration_s=20)
