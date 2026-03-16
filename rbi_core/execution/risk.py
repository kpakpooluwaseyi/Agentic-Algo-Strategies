"""rbi_core/execution/risk.py — Dynamic position sizing + execution queue."""
import threading
import queue
import time
from typing import Optional


class PortfolioExecutionQueue:
    """
    Thread-safe execution queue.
    Position sizing uses LIVE account equity, not static initial value.
    """

    def __init__(self, initial_equity: float, max_concurrent: int = 5,
                 risk_per_trade_pct: float = 0.01):
        """
        Args:
            initial_equity: Starting account size in USD.
            max_concurrent: Max simultaneous open positions.
            risk_per_trade_pct: Fraction of equity risked per trade (0.01 = 1%).
        """
        self.trade_queue: queue.Queue = queue.Queue()
        self._equity: float = initial_equity
        self._equity_lock = threading.Lock()
        self.active_positions: dict[str, dict] = {}
        self._positions_lock = threading.Lock()
        self.max_concurrent = max_concurrent
        self.risk_per_trade_pct = risk_per_trade_pct
        self.halted: bool = False  # NanoClaw sets this

    @property
    def equity(self) -> float:
        with self._equity_lock:
            return self._equity

    def update_equity(self, new_equity: float) -> None:
        """Called after each trade close or periodic balance fetch."""
        with self._equity_lock:
            self._equity = new_equity

    def halt(self) -> None:
        """Emergency halt. Called by NanoClaw watchdog."""
        self.halted = True

    def resume(self) -> None:
        self.halted = False

    def calculate_position_size(self, current_atr: float, atr_multiplier: float) -> float:
        """
        Dynamic position sizing: risk_per_trade_pct of CURRENT equity.
        Returns size in units (contracts/coins).
        """
        risk_amount = self.equity * self.risk_per_trade_pct
        stop_distance = current_atr * atr_multiplier
        if stop_distance <= 0:
            return 0.0
        return risk_amount / stop_distance

    def submit_signal(self, signal: dict, params: dict,
                      market_data: dict) -> Optional[str]:
        """
        Validate and queue a trade signal.

        Args:
            signal: {'action': 'BUY'|'SELL', 'strategy': str, ...}
            params: Active RL params (must contain 'atr_multiplier').
            market_data: {'price': float, 'atr': float, ...}

        Returns:
            position_key if queued, else None.
        """
        if self.halted:
            return None

        with self._positions_lock:
            if len(self.active_positions) >= self.max_concurrent:
                return None

            pos_size = self.calculate_position_size(
                market_data['atr'],
                params['atr_multiplier'],
            )
            if pos_size <= 0:
                return None

            position_key = signal.get('position_key') or f"{signal.get('strategy', 'unknown')}_{time.time_ns()}"

            order = {
                'action': signal['action'],
                'size': pos_size,
                'strategy': signal.get('strategy', 'unknown'),
                'price': market_data['price'],
                'timestamp': time.time(),
                'position_key': position_key,
            }
            self.trade_queue.put(order)
            self.active_positions[position_key] = order
            return position_key

    def close_position(self, position_key: str, realized_pnl: float) -> None:
        """Remove position and update equity."""
        with self._positions_lock:
            self.active_positions.pop(position_key, None)
        self.update_equity(self.equity + realized_pnl)
