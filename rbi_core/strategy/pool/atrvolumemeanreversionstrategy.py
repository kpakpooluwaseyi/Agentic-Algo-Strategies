from typing import Optional
from collections import deque
from rbi_core.strategy.base import BaseStrategy, Signal

class ATRVolumeMeanReversionStrategy(BaseStrategy):
    def __init__(self, window: int = 20, atr_multiplier: float = 1.5, volume_boost: float = 0.15):
        self.window = window
        self.atr_multiplier = atr_multiplier
        self.volume_boost = volume_boost
        self.reset()

    def reset(self) -> None:
        self.prices = deque(maxlen=self.window)
        self.volumes = deque(maxlen=self.window)

    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        volume = tick_data.volume

        if len(self.prices) < self.window:
            self.prices.append(price)
            self.volumes.append(volume)
            return None

        sma = sum(self.prices) / self.window
        avg_volume = sum(self.volumes) / self.window
        upper_band = sma + self.atr_multiplier * atr
        lower_band = sma - self.atr_multiplier * atr

        action = 'HOLD'
        confidence = 0.0

        if price > upper_band:
            action = 'SELL'
            raw_deviation = (price - upper_band) / atr if atr > 0 else 0.0
            base_confidence = min(0.5 + raw_deviation * 0.1, 0.85)
            confidence = min(base_confidence + self.volume_boost, 1.0) if volume > avg_volume else base_confidence
        elif price < lower_band:
            action = 'BUY'
            raw_deviation = (lower_band - price) / atr if atr > 0 else 0.0
            base_confidence = min(0.5 + raw_deviation * 0.1, 0.85)
            confidence = min(base_confidence + self.volume_boost, 1.0) if volume > avg_volume else base_confidence

        meta = {
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'avg_volume': avg_volume,
            'current_price': price,
            'current_volume': volume,
            'atr': atr
        }

        self.prices.append(price)
        self.volumes.append(volume)

        return Signal(action=action, confidence=confidence, meta=meta)