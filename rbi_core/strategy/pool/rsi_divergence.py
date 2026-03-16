"""rbi_core/strategy/pool/rsi_divergence.py — RSI Divergence reversal strategy."""
from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque
import numpy as np


class RSIDivergence(BaseStrategy):
    """
    Detects RSI divergence against price.
    - Bullish divergence: price makes lower low, RSI makes higher low → BUY
    - Bearish divergence: price makes higher high, RSI makes lower high → SELL

    Accumulates ticks into bars internally, computes RSI, then checks for divergence.
    """

    def __init__(self, rsi_period: int = 14, bar_ticks: int = 50,
                 divergence_lookback: int = 5, overbought: float = 70.0,
                 oversold: float = 30.0):
        super().__init__(name="RSI_Divergence")
        self.rsi_period = rsi_period
        self.bar_ticks = bar_ticks  # aggregate N ticks into one bar
        self.divergence_lookback = divergence_lookback
        self.overbought = overbought
        self.oversold = oversold

        # Internal state
        self._tick_count = 0
        self._bar_prices: deque = deque(maxlen=rsi_period + divergence_lookback + 5)
        self._bar_highs: deque = deque(maxlen=divergence_lookback + 2)
        self._bar_lows: deque = deque(maxlen=divergence_lookback + 2)
        self._rsi_values: deque = deque(maxlen=divergence_lookback + 2)
        self._current_bar_ticks: list = []

    def on_tick(self, tick_data: dict) -> Optional[Signal]:
        self._tick_count += 1
        self._current_bar_ticks.append(tick_data['price'])

        # Accumulate ticks into a bar
        if len(self._current_bar_ticks) < self.bar_ticks:
            return None

        # Close the bar
        bar_close = self._current_bar_ticks[-1]
        bar_high = max(self._current_bar_ticks)
        bar_low = min(self._current_bar_ticks)
        self._current_bar_ticks = []

        self._bar_prices.append(bar_close)
        self._bar_highs.append(bar_high)
        self._bar_lows.append(bar_low)

        # Need enough bars to compute RSI
        if len(self._bar_prices) < self.rsi_period + 1:
            return None

        rsi = self._compute_rsi()
        self._rsi_values.append(rsi)

        if len(self._rsi_values) < 2 or len(self._bar_lows) < 2:
            return None

        # Check for bullish divergence (price lower low, RSI higher low)
        if (self._bar_lows[-1] < self._bar_lows[-2] and
                self._rsi_values[-1] > self._rsi_values[-2] and
                rsi < self.oversold + 10):
            self.current_confidence = min(1.0, (self.oversold + 10 - rsi) / 20.0)
            return Signal(action="BUY", confidence=self.current_confidence,
                          meta={"trigger": "bullish_divergence", "rsi": rsi})

        # Check for bearish divergence (price higher high, RSI lower high)
        if (self._bar_highs[-1] > self._bar_highs[-2] and
                self._rsi_values[-1] < self._rsi_values[-2] and
                rsi > self.overbought - 10):
            self.current_confidence = min(1.0, (rsi - self.overbought + 10) / 20.0)
            return Signal(action="SELL", confidence=self.current_confidence,
                          meta={"trigger": "bearish_divergence", "rsi": rsi})

        self.current_confidence = 0.0
        return None

    def _compute_rsi(self) -> float:
        """Compute RSI from accumulated bar close prices."""
        prices = list(self._bar_prices)
        if len(prices) < self.rsi_period + 1:
            return 50.0

        deltas = np.diff(prices[-(self.rsi_period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def reset(self) -> None:
        self._tick_count = 0
        self._bar_prices.clear()
        self._bar_highs.clear()
        self._bar_lows.clear()
        self._rsi_values.clear()
        self._current_bar_ticks = []
        self.current_confidence = 0.0
