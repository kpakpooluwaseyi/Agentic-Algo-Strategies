"""rbi_core/data/atr_calculator.py — Rolling Wilder ATR from tick-aggregated bars.

Solves the P0 'atr: 0.0' problem. Accumulates raw ticks into bars, computes
True Range per bar, and applies Wilder's smoothed EMA to produce a live ATR value.

Thread-safe: written by WS thread, read by combiner/risk threads.
"""
import threading
from collections import deque
from typing import Optional


class ATRCalculator:
    """
    Computes Average True Range from streaming tick data.

    1. Accumulates `bar_ticks` ticks into a single OHLC bar.
    2. Computes True Range: max(H-L, |H-prev_close|, |L-prev_close|).
    3. Applies Wilder's smoothed EMA over `atr_period` bars.

    After warmup (atr_period bars completed), `self.atr` is always > 0.
    """

    def __init__(self, bar_ticks: int = 50, atr_period: int = 14):
        """
        Args:
            bar_ticks: Number of ticks to aggregate into one bar.
            atr_period: Wilder ATR lookback in bars (default 14).
        """
        self.bar_ticks = bar_ticks
        self.atr_period = atr_period

        # Bar accumulation state
        self._current_bar_prices: list[float] = []
        self._prev_close: float = 0.0

        # ATR state (thread-safe)
        self._lock = threading.Lock()
        self._true_ranges: deque = deque(maxlen=atr_period + 1)
        self._atr: float = 0.0
        self._bar_count: int = 0
        self._warmed_up: bool = False

    @property
    def atr(self) -> float:
        """Current ATR value. Returns 0.0 before warmup completes."""
        with self._lock:
            return self._atr

    @property
    def is_warmed_up(self) -> bool:
        with self._lock:
            return self._warmed_up

    def process_tick(self, tick: dict) -> None:
        """
        Feed a tick into the ATR calculator. Accumulates into bars internally.

        Args:
            tick: Must contain 'price' key at minimum.
        """
        price = tick.get('price', 0.0)
        if price <= 0:
            return

        self._current_bar_prices.append(price)

        if len(self._current_bar_prices) < self.bar_ticks:
            return

        # Close the bar
        bar_high = max(self._current_bar_prices)
        bar_low = min(self._current_bar_prices)
        bar_close = self._current_bar_prices[-1]
        self._current_bar_prices = []

        # Compute True Range
        if self._prev_close > 0:
            true_range = max(
                bar_high - bar_low,
                abs(bar_high - self._prev_close),
                abs(bar_low - self._prev_close),
            )
        else:
            true_range = bar_high - bar_low

        self._prev_close = bar_close

        with self._lock:
            self._true_ranges.append(true_range)
            self._bar_count += 1
            self._update_atr()

    def _update_atr(self) -> None:
        """Wilder's smoothed ATR: ATR = ((prev_ATR * (N-1)) + TR) / N"""
        n = self.atr_period
        if self._bar_count < n:
            return  # Not enough bars yet

        if not self._warmed_up:
            # First ATR = simple average of first N true ranges
            trs = list(self._true_ranges)
            self._atr = sum(trs[:n]) / n
            self._warmed_up = True
        else:
            # Wilder's smoothing
            current_tr = self._true_ranges[-1]
            self._atr = ((self._atr * (n - 1)) + current_tr) / n

    def reset(self) -> None:
        """Reset all state. Used on session boundary or strategy reset."""
        with self._lock:
            self._current_bar_prices = []
            self._prev_close = 0.0
            self._true_ranges.clear()
            self._atr = 0.0
            self._bar_count = 0
            self._warmed_up = False
