from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class ATRVWAPMeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback_period: int = 20, atr_deviation_threshold: float = 1.5, max_spread_atr_ratio: float = 0.2):
        super().__init__()
        self.lookback_period = lookback_period
        self.atr_deviation_threshold = atr_deviation_threshold
        self.max_spread_atr_ratio = max_spread_atr_ratio
        self.price_volume_products = deque(maxlen=lookback_period)
        self.volumes = deque(maxlen=lookback_period)
        
    def reset(self) -> None:
        self.price_volume_products.clear()
        self.volumes.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        bid = tick_data.bid
        ask = tick_data.ask
        
        if atr <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={'error': 'invalid_atr'})
            
        current_spread = ask - bid
        if current_spread / atr > self.max_spread_atr_ratio:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'excessive_spread', 'spread_atr_ratio': current_spread / atr})
        
        self.price_volume_products.append(price * volume)
        self.volumes.append(volume)
        
        if len(self.volumes) < self.lookback_period:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'insufficient_data', 'collected': len(self.volumes), 'required': self.lookback_period})
        
        total_volume = sum(self.volumes)
        if total_volume == 0:
            return Signal(action='HOLD', confidence=0.0, meta={'error': 'zero_total_volume'})
            
        vwap = sum(self.price_volume_products) / total_volume
        deviation = (price - vwap) / atr
        
        if deviation > self.atr_deviation_threshold:
            confidence = min((deviation - self.atr_deviation_threshold) / self.atr_deviation_threshold, 1.0)
            return Signal(
                action='SELL',
                confidence=confidence,
                meta={
                    'strategy': 'vwap_mean_reversion',
                    'signal_type': 'overbought',
                    'vwap': vwap,
                    'deviation_atr': deviation,
                    'spread': current_spread
                }
            )
        elif deviation < -self.atr_deviation_threshold:
            confidence = min((abs(deviation) - self.atr_deviation_threshold) / self.atr_deviation_threshold, 1.0)
            return Signal(
                action='BUY',
                confidence=confidence,
                meta={
                    'strategy': 'vwap_mean_reversion',
                    'signal_type': 'oversold',
                    'vwap': vwap,
                    'deviation_atr': deviation,
                    'spread': current_spread
                }
            )
        
        return Signal(action='HOLD', confidence=0.0, meta={'vwap': vwap, 'deviation_atr': deviation, 'mid_price': (bid + ask) / 2})