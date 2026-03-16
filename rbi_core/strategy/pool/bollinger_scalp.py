"""rbi_core/strategy/pool/bollinger_scalp.py — Bollinger Band Scalp strategy."""
from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque
import numpy as np


class BollingerScalp(BaseStrategy):
    """
    Bollinger Band mean-reversion scalper.
    - BUY when price touches lower band (price < SMA - N*std)
    - SELL when price touches upper band (price > SMA + N*std)

    Operates on accumulated ticks aggregated into bars.
    """

    def __init__(self, period: int = 20, num_std: float = 2.0,
                 bar_ticks: int = 30, band_touch_margin: float = 0.0005):
        super().__init__(name="Bollinger_Scalp")
        self.period = period
        self.num_std = num_std
        self.bar_ticks = bar_ticks
        self.band_touch_margin = band_touch_margin

        self._current_bar: list = []
        self._closes: deque = deque(maxlen=period + 1)

    def on_tick(self, tick_data: dict) -> Optional[Signal]:
        self._current_bar.append(tick_data['price'])

        if len(self._current_bar) < self.bar_ticks:
            return None

        # Close the bar
        bar_close = self._current_bar[-1]
        self._current_bar = []
        self._closes.append(bar_close)

        if len(self._closes) < self.period:
            return None

        closes = list(self._closes)[-self.period:]
        sma = np.mean(closes)
        std = np.std(closes)

        if std == 0:
            return None

        upper_band = sma + self.num_std * std
        lower_band = sma - self.num_std * std

        # BUY: price at or below lower band
        if bar_close <= lower_band * (1 + self.band_touch_margin):
            z_score = abs((bar_close - sma) / std)
            self.current_confidence = min(1.0, z_score / (self.num_std * 1.5))
            return Signal(
                action="BUY",
                confidence=self.current_confidence,
                meta={"trigger": "lower_band_touch", "sma": sma,
                       "lower": lower_band, "upper": upper_band}
            )

        # SELL: price at or above upper band
        if bar_close >= upper_band * (1 - self.band_touch_margin):
            z_score = abs((bar_close - sma) / std)
            self.current_confidence = min(1.0, z_score / (self.num_std * 1.5))
            return Signal(
                action="SELL",
                confidence=self.current_confidence,
                meta={"trigger": "upper_band_touch", "sma": sma,
                       "lower": lower_band, "upper": upper_band}
            )

        self.current_confidence = 0.0
        return None

    def reset(self) -> None:
        self._current_bar = []
        self._closes.clear()
        self.current_confidence = 0.0
