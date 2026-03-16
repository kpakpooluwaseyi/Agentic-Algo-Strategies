from typing import Optional, List
from rbi_core.strategy.base import BaseStrategy, Signal

class ATRBandMeanReversionStrategy(BaseStrategy):
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        self.price_history: List[float] = []
        self.lookback = 20
        self.last_signal = "HOLD"
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        
        self.price_history.append(price)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)
        
        if len(self.price_history) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'initializing'})
        
        sma = sum(self.price_history) / self.lookback
        upper_band = sma + 2.0 * atr
        lower_band = sma - 2.0 * atr
        
        if price > upper_band:
            deviation = (price - upper_band) / atr if atr > 0 else 0
            confidence = min(1.0, 0.4 + (deviation * 0.15))
            self.last_signal = "SELL"
            return Signal(
                action='SELL',
                confidence=confidence,
                meta={'sma': sma, 'upper': upper_band, 'deviation': deviation}
            )
        elif price < lower_band:
            deviation = (lower_band - price) / atr if atr > 0 else 0
            confidence = min(1.0, 0.4 + (deviation * 0.15))
            self.last_signal = "BUY"
            return Signal(
                action='BUY',
                confidence=confidence,
                meta={'sma': sma, 'lower': lower_band, 'deviation': deviation}
            )
        
        return Signal(action='HOLD', confidence=0.0, meta={'sma': sma, 'band': 'inside'})