from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class ATRChannelMeanReversionStrategy(BaseStrategy):
    def __init__(self, period: int = 20, multiplier: float = 2.0):
        super().__init__()
        self.period = period
        self.multiplier = multiplier
        self.prices = deque(maxlen=period)
        self.last_signal = 'HOLD'

    def reset(self) -> None:
        self.prices.clear()
        self.last_signal = 'HOLD'

    def on_tick(self, tick_data) -> Optional[Signal]:
        current_price = tick_data.price
        current_atr = tick_data.atr
        
        self.prices.append(current_price)
        
        if len(self.prices) < self.period:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'initializing', 'samples': len(self.prices)})
        
        sma = sum(self.prices) / self.period
        upper_band = sma + self.multiplier * current_atr
        lower_band = sma - self.multiplier * current_atr
        
        if current_price > upper_band:
            distance = current_price - upper_band
            confidence = min(1.0, distance / (current_atr + 1e-9))
            return Signal(action='SELL', confidence=confidence, meta={
                'sma': sma,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'atr': current_atr,
                'distance': distance,
                'logic': 'mean_reversion_overbought'
            })
        elif current_price < lower_band:
            distance = lower_band - current_price
            confidence = min(1.0, distance / (current_atr + 1e-9))
            return Signal(action='BUY', confidence=confidence, meta={
                'sma': sma,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'atr': current_atr,
                'distance': distance,
                'logic': 'mean_reversion_oversold'
            })
        else:
            return Signal(action='HOLD', confidence=0.0, meta={
                'sma': sma,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'atr': current_atr,
                'position_in_channel': (current_price - lower_band) / (upper_band - lower_band + 1e-9),
                'logic': 'within_normal_range'
            })