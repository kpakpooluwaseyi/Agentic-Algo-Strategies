"""rbi_core/execution/dispatcher.py — Order dispatch to exchange API (mock/live)."""
import threading
import time
from typing import Optional


class OrderDispatcher:
    """
    Consumes orders from PortfolioExecutionQueue.trade_queue.
    In mock mode: simulates fills. In live mode: hits exchange API.
    """

    def __init__(self, execution_queue, mode: str = "mock"):
        """
        Args:
            execution_queue: PortfolioExecutionQueue instance.
            mode: "mock" for paper trading, "live" for real execution.
        """
        self.queue = execution_queue
        self.mode = mode
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _dispatch_loop(self) -> None:
        while self._running:
            try:
                order = self.queue.trade_queue.get(timeout=1.0)
                if self.queue.halted:
                    print(f"[Dispatcher] HALTED — dropping order: {order}")
                    continue
                self._execute(order)
            except Exception:
                pass  # queue.Empty on timeout

    def _execute(self, order: dict) -> None:
        if self.mode == "mock":
            print(f"[MockExec] {order['action']} {order['size']:.4f} @ {order['price']:.2f} "
                  f"({order['strategy']})")
        else:
            # TODO: Implement live exchange API call
            raise NotImplementedError("Live execution not implemented")
