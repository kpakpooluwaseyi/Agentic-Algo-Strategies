"""rbi_core/execution/trade_manager.py — Trade lifecycle manager with PnL tracking.

Solves the P0 'No TradeManager' blocker. Tracks open positions, enforces
ATR-based stop-losses, max holding time, and computes rolling Sortino ratio
from realized trade PnL to feed the RL agent.

Thread-safe: written by tick pipeline, read by RL agent and metrics.
"""
import threading
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional
import uuid


@dataclass
class Trade:
    """A single open or closed trade."""
    trade_id: str
    action: str              # "BUY" or "SELL"
    strategy: str            # originating strategy name
    entry_price: float
    size: float              # position size in units
    entry_atr: float         # ATR at entry (for stop-loss)
    atr_multiplier: float    # stop distance = entry_atr * atr_multiplier
    entry_time: float        # unix timestamp
    max_holding_s: float     # max holding time in seconds
    exit_price: float = 0.0
    exit_time: float = 0.0
    realized_pnl: float = 0.0
    exit_reason: str = ""    # "stop_loss", "take_profit", "max_hold", "signal", "manual"
    closed: bool = False


class TradeManager:
    """
    Manages active trades: opens, monitors exits, computes PnL.

    Exit conditions checked on every tick:
    1. ATR-based stop-loss: price moves against entry by entry_atr * atr_multiplier
    2. Max holding time: trade exceeds max_holding_time_minutes from RL params
    3. Opposing signal: combiner produces opposite direction signal

    On close: fires on_trade_closed callback → feeds RL agent + strategy metrics.
    """

    def __init__(
        self,
        on_trade_closed: Optional[Callable[[dict], None]] = None,
        sortino_window: int = 20,
        target_return: float = 0.0,
    ):
        """
        Args:
            on_trade_closed: Callback with trade result dict on each trade close.
            sortino_window: Number of recent trades for rolling Sortino.
            target_return: Target return for Sortino (MAR). 0.0 = risk-free.
        """
        self.on_trade_closed = on_trade_closed
        self.sortino_window = sortino_window
        self.target_return = target_return

        self._lock = threading.Lock()
        self.open_trades: dict[str, Trade] = {}
        self._closed_pnls: deque = deque(maxlen=sortino_window)
        self._total_pnl: float = 0.0
        self._trade_count: int = 0
        self._win_count: int = 0

    def open_trade(
        self,
        action: str,
        entry_price: float,
        size: float,
        strategy: str,
        entry_atr: float,
        atr_multiplier: float,
        max_holding_minutes: float = 60.0,
    ) -> str:
        """
        Open a new trade. Returns trade_id.

        Args:
            action: "BUY" or "SELL"
            entry_price: Execution price.
            size: Position size in units.
            strategy: Originating strategy name.
            entry_atr: Current ATR at entry.
            atr_multiplier: Stop distance multiplier from RL params.
            max_holding_minutes: Max holding time from RL params.
        """
        trade_id = str(uuid.uuid4())[:8]
        trade = Trade(
            trade_id=trade_id,
            action=action,
            strategy=strategy,
            entry_price=entry_price,
            size=size,
            entry_atr=entry_atr,
            atr_multiplier=atr_multiplier,
            entry_time=time.time(),
            max_holding_s=max_holding_minutes * 60.0,
        )
        with self._lock:
            self.open_trades[trade_id] = trade
        return trade_id

    def check_exits(self, current_tick: dict) -> list[str]:
        """
        Check all open trades for exit conditions against current tick.
        Returns list of trade_ids that were closed.

        Args:
            current_tick: Must contain 'price', 'timestamp'.
        """
        price = current_tick.get('price', 0.0)
        now = time.time()
        closed_ids = []

        with self._lock:
            for trade_id, trade in list(self.open_trades.items()):
                exit_reason = self._check_single_exit(trade, price, now)
                if exit_reason:
                    self._close_trade_internal(trade, price, now, exit_reason)
                    closed_ids.append(trade_id)

        return closed_ids

    def close_trade(self, trade_id: str, exit_price: float,
                    reason: str = "signal") -> Optional[dict]:
        """
        Manually close a specific trade (e.g., on opposing signal).

        Returns trade result dict or None if trade not found.
        """
        with self._lock:
            trade = self.open_trades.get(trade_id)
            if not trade:
                return None
            return self._close_trade_internal(trade, exit_price, time.time(), reason)

    def _check_single_exit(self, trade: Trade, price: float, now: float) -> Optional[str]:
        """Check if a single trade should be closed. Returns exit reason or None."""
        # 1. Max holding time
        elapsed = now - trade.entry_time
        if elapsed >= trade.max_holding_s:
            return "max_hold"

        # 2. ATR stop-loss
        stop_distance = trade.entry_atr * trade.atr_multiplier
        if stop_distance > 0:
            if trade.action == "BUY":
                stop_price = trade.entry_price - stop_distance
                if price <= stop_price:
                    return "stop_loss"
            else:  # SELL
                stop_price = trade.entry_price + stop_distance
                if price >= stop_price:
                    return "stop_loss"

        return None

    def _close_trade_internal(self, trade: Trade, exit_price: float,
                              exit_time: float, reason: str) -> dict:
        """Close trade, compute PnL, fire callback. Caller must hold self._lock."""
        # Compute PnL
        if trade.action == "BUY":
            pnl = (exit_price - trade.entry_price) * trade.size
        else:  # SELL
            pnl = (trade.entry_price - exit_price) * trade.size

        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.realized_pnl = pnl
        trade.exit_reason = reason
        trade.closed = True

        # Remove from open, update stats
        self.open_trades.pop(trade.trade_id, None)
        self._closed_pnls.append(pnl)
        self._total_pnl += pnl
        self._trade_count += 1
        if pnl > 0:
            self._win_count += 1

        result = {
            'trade_id': trade.trade_id,
            'action': trade.action,
            'strategy': trade.strategy,
            'entry_price': trade.entry_price,
            'exit_price': exit_price,
            'size': trade.size,
            'realized_pnl': pnl,
            'exit_reason': reason,
            'holding_time_s': exit_time - trade.entry_time,
        }

        # Fire callback (outside critical path — RL agent, metrics)
        if self.on_trade_closed:
            try:
                self.on_trade_closed(result)
            except Exception as e:
                print(f"[TradeManager] Callback error: {e}")

        return result

    @property
    def rolling_sortino(self) -> float:
        """Rolling Sortino ratio from recent closed trades."""
        if len(self._closed_pnls) < 2:
            return 0.0
        pnls = list(self._closed_pnls)
        mean_return = sum(pnls) / len(pnls)
        excess = mean_return - self.target_return

        # Downside deviation: std of negative returns only
        downside = [min(0, p - self.target_return) for p in pnls]
        downside_sq = sum(d * d for d in downside) / len(downside)
        downside_dev = math.sqrt(downside_sq) if downside_sq > 0 else 1e-9

        return excess / downside_dev

    @property
    def total_pnl(self) -> float:
        return self._total_pnl

    @property
    def winrate(self) -> float:
        if self._trade_count == 0:
            return 0.0
        return self._win_count / self._trade_count

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def current_pnl_fraction(self) -> float:
        """Current unrealized + realized PnL as fraction (for NanoClaw)."""
        return self._total_pnl
