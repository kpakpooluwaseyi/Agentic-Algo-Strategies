"""rbi_core/swarm_integration/nanoclaw_monitor.py — Deterministic local kill-switch watchdog."""
import threading
import time
from typing import Callable, Optional


class NanoClawMonitor:
    """
    Lightweight local watchdog daemon. No LLM. Pure threshold checks.
    Independently monitors system health and triggers emergency halt.

    Monitors:
    1. Tick staleness (no new tick for N seconds = feed dead)
    2. Spread collapse (ask - bid < threshold = flash crash / no liquidity)
    3. PnL breach (account drawdown exceeds limit)
    4. Heartbeat to Dell swarm (optional secondary check)
    """

    def __init__(
        self,
        halt_callback: Callable[[], None],           # PortfolioExecutionQueue.halt
        get_last_tick_ts: Callable[[], float],        # Returns timestamp of latest tick
        get_current_spread: Callable[[], float],      # Returns current bid-ask spread
        get_current_pnl: Callable[[], float],         # Returns current PnL as fraction
        check_interval_s: float = 1.0,                # Check every 1 second
        tick_stale_threshold_s: float = 10.0,         # No tick for 10s = stale
        min_spread: float = 0.0001,                   # Spread below this = danger
        pnl_kill_threshold: float = -0.08,            # -8% account drawdown = halt
    ):
        self.halt_callback = halt_callback
        self.get_last_tick_ts = get_last_tick_ts
        self.get_current_spread = get_current_spread
        self.get_current_pnl = get_current_pnl
        self.check_interval_s = check_interval_s
        self.tick_stale_threshold_s = tick_stale_threshold_s
        self.min_spread = min_spread
        self.pnl_kill_threshold = pnl_kill_threshold
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._halt_triggered = False

    def start(self) -> None:
        self._running = True
        self._halt_triggered = False
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[NanoClaw] Watchdog started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        print("[NanoClaw] Watchdog stopped")

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                self._check_all()
            except Exception as e:
                # NanoClaw must NEVER crash. Log and continue.
                print(f"[NanoClaw] Internal error (non-fatal): {e}")
            time.sleep(self.check_interval_s)

    def _check_all(self) -> None:
        if self._halt_triggered:
            return  # Already halted, don't spam

        # Check 1: Tick staleness
        last_tick = self.get_last_tick_ts()
        tick_age = time.time() - last_tick
        if tick_age > self.tick_stale_threshold_s:
            self._trigger_halt(f"TICK STALE: No tick for {tick_age:.1f}s")
            return

        # Check 2: Spread collapse
        spread = self.get_current_spread()
        if spread < self.min_spread:
            self._trigger_halt(f"SPREAD COLLAPSE: spread={spread:.6f}")
            return

        # Check 3: PnL breach
        pnl = self.get_current_pnl()
        if pnl < self.pnl_kill_threshold:
            self._trigger_halt(f"PNL BREACH: pnl={pnl:.4f} < kill={self.pnl_kill_threshold}")
            return

    def _trigger_halt(self, reason: str) -> None:
        print(f"[NanoClaw] *** EMERGENCY HALT *** Reason: {reason}")
        self._halt_triggered = True
        self.halt_callback()
