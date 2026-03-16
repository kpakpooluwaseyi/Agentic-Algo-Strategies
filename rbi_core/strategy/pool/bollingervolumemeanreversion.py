from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class BollingerVolumeMeanReversion(BaseStrategy):
    def __init__(self, lookback: int = 20, bb_mult: float = 2.0, vol_lookback: int = 20):
        super().__init__()
        self.lookback = lookback
        self.bb_mult = bb_mult
        self.vol_lookback = vol_lookback
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=vol_lookback)
        self.atrs = deque(maxlen=lookback)
        
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.atrs.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        self.prices.append(price)
        self.volumes.append(volume)
        self.atrs.append(atr)
        
        if len(self.prices) < self.lookback:
            return None
            
        sma = sum(self.prices) / len(self.prices)
        variance = sum((p - sma) ** 2 for p in self.prices) / len(self.prices)
        std_dev = variance ** 0.5
        
        upper_band = sma + (self.bb_mult * std_dev)
        lower_band = sma - (self.bb_mult * std_dev)
        avg_volume = sum(self.volumes) / len(self.volumes)
        avg_atr = sum(self.atrs) / len(self.atrs)
        
        action = 'HOLD'
        confidence = 0.0
        deviation = 0.0
        
        if price < lower_band and volume > avg_volume * 1.3 and atr < avg_atr * 1.5:
            action = 'BUY'
            deviation = (lower_band - price) / std_dev if std_dev > 0 else 0
            confidence = min(0.5 + deviation * 0.15, 0.9)
        elif price > upper_band and volume > avg_volume * 1.3 and atr < avg_atr * 1.5:
            action = 'SELL'
            deviation = (price - upper_band) / std_dev if std_dev > 0 else 0
            confidence = min(0.5 + deviation * 0.15, 0.9)
            
        if action == 'HOLD':
            return None
            
        return Signal(action=action, confidence=round(confidence, 4), meta={
            'sma': round(sma, 4),
            'upper': round(upper_band, 4),
            'lower': round(lower_band, 4),
            'vol_ratio': round(volume / avg_volume, 2),
            'deviation': round(deviation, 4)
        })