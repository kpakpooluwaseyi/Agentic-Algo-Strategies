"""rbi_core/strategy/pool/vwap_mean_reversion.py — VWAP Mean Reversion strategy."""
from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque


class VWAPMeanReversion(BaseStrategy):
    """
    Mean reversion around intraday VWAP.
    - BUY when price drops below VWAP by `entry_deviation_pct`
    - SELL when price rises above VWAP by `entry_deviation_pct`

    VWAP is computed from accumulated tick-level price * volume data.
    Resets daily or after `reset_ticks` ticks (simulates session boundary).
    """

    def __init__(self, entry_deviation_pct: float = 0.3,
                 exit_deviation_pct: float = 0.05,
                 min_ticks_for_vwap: int = 100,
                 reset_ticks: int = 10000):
        super().__init__(name="VWAP_MeanReversion")
        self.entry_deviation_pct = entry_deviation_pct / 100.0
        self.exit_deviation_pct = exit_deviation_pct / 100.0
        self.min_ticks_for_vwap = min_ticks_for_vwap
        self.reset_ticks = reset_ticks

        # VWAP state
        self._cumulative_pv: float = 0.0  # sum(price * volume)
        self._cumulative_vol: float = 0.0  # sum(volume)
        self._tick_count: int = 0
        self._last_signal: Optional[str] = None

    def on_tick(self, tick_data: dict) -> Optional[Signal]:
        price = tick_data['price']
        volume = tick_data.get('volume', 0.0)

        # Accumulate VWAP
        if volume > 0:
            self._cumulative_pv += price * volume
            self._cumulative_vol += volume
        self._tick_count += 1

        # Auto-reset after N ticks (session boundary proxy)
        if self._tick_count >= self.reset_ticks:
            self.reset()
            return None

        # Need minimum data before generating signals
        if self._tick_count < self.min_ticks_for_vwap or self._cumulative_vol == 0:
            return None

        vwap = self._cumulative_pv / self._cumulative_vol
        deviation = (price - vwap) / vwap

        # BUY: price significantly below VWAP
        if deviation < -self.entry_deviation_pct and self._last_signal != "BUY":
            self.current_confidence = min(1.0, abs(deviation) / (self.entry_deviation_pct * 2))
            self._last_signal = "BUY"
            return Signal(
                action="BUY",
                confidence=self.current_confidence,
                meta={"trigger": "below_vwap", "vwap": vwap, "deviation_pct": deviation * 100}
            )

        # SELL: price significantly above VWAP
        if deviation > self.entry_deviation_pct and self._last_signal != "SELL":
            self.current_confidence = min(1.0, abs(deviation) / (self.entry_deviation_pct * 2))
            self._last_signal = "SELL"
            return Signal(
                action="SELL",
                confidence=self.current_confidence,
                meta={"trigger": "above_vwap", "vwap": vwap, "deviation_pct": deviation * 100}
            )

        # Neutralize signal if price returns near VWAP
        if abs(deviation) < self.exit_deviation_pct:
            self._last_signal = None

        self.current_confidence = 0.0
        return None

    def reset(self) -> None:
        self._cumulative_pv = 0.0
        self._cumulative_vol = 0.0
        self._tick_count = 0
        self._last_signal = None
        self.current_confidence = 0.0
