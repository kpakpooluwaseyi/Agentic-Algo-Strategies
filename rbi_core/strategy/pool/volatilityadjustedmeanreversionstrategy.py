from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque


class VolatilityAdjustedMeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, deviation_threshold: float = 1.5, min_atr: float = 1e-6):
        super().__init__()
        self.lookback = lookback
        self.deviation_threshold = deviation_threshold
        self.min_atr = min_atr
        self.price_history = deque(maxlen=lookback)
        self.mid_price_history = deque(maxlen=lookback)
        
    def reset(self) -> None:
        self.price_history.clear()
        self.mid_price_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        bid = tick_data.bid
        ask = tick_data.ask
        
        mid_price = (bid + ask) / 2.0
        
        self.price_history.append(price)
        self.mid_price_history.append(mid_price)
        
        if len(self.price_history) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'insufficient_history', 'count': len(self.price_history)})
        
        if atr < self.min_atr:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'atr_below_minimum', 'atr': atr})
        
        current_deviation = price - mid_price
        normalized_deviation = current_deviation / atr if atr > 0 else 0.0
        
        action = 'HOLD'
        confidence = 0.0
        meta = {
            'price': price,
            'mid_price': mid_price,
            'atr': atr,
            'deviation': current_deviation,
            'normalized_deviation': normalized_deviation,
            'threshold': self.deviation_threshold
        }
        
        if normalized_deviation > self.deviation_threshold:
            action = 'SELL'
            confidence = min(1.0, abs(normalized_deviation) / (self.deviation_threshold * 2.0))
            meta['signal_type'] = 'overbought_reversion'
        elif normalized_deviation < -self.deviation_threshold:
            action = 'BUY'
            confidence = min(1.0, abs(normalized_deviation) / (self.deviation_threshold * 2.0))
            meta['signal_type'] = 'oversold_reversion'
        else:
            confidence = max(0.0, abs(normalized_deviation) / self.deviation_threshold - 0.5)
            meta['signal_type'] = 'within_threshold'
            
        return Signal(action=action, confidence=float(confidence), meta=meta)