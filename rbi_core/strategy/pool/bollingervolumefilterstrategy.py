from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional, List
import statistics

class BollingerVolumeFilterStrategy(BaseStrategy):
    def __init__(self, window: int = 20, num_std: float = 2.0, volume_multiplier: float = 1.2, max_spread_pct: float = 0.001, min_atr_ratio: float = 0.9):
        super().__init__()
        self.window = window
        self.num_std = num_std
        self.volume_multiplier = volume_multiplier
        self.max_spread_pct = max_spread_pct
        self.min_atr_ratio = min_atr_ratio
        self.prices: List[float] = []
        self.volumes: List[float] = []
        self.atrs: List[float] = []
        
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.atrs.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        bid = tick_data.bid
        ask = tick_data.ask
        timestamp = tick_data.timestamp
        
        self.prices.append(price)
        self.volumes.append(volume)
        self.atrs.append(atr)
        
        if len(self.prices) > self.window:
            self.prices.pop(0)
        if len(self.volumes) > self.window:
            self.volumes.pop(0)
        if len(self.atrs) > self.window:
            self.atrs.pop(0)
            
        if len(self.prices) < self.window:
            return Signal(action='HOLD', confidence=0.0, meta={'timestamp': timestamp, 'reason': 'insufficient_data'})
            
        sma = statistics.mean(self.prices)
        std_dev = statistics.stdev(self.prices) if len(set(self.prices)) > 1 else 0.001
        upper_band = sma + (self.num_std * std_dev)
        lower_band = sma - (self.num_std * std_dev)
        
        avg_volume = statistics.mean(self.volumes)
        avg_atr = statistics.mean(self.atrs)
        
        spread_pct = (ask - bid) / price if price > 0 else 1.0
        
        meta = {
            'timestamp': timestamp,
            'price': price,
            'sma': round(sma, 4),
            'upper_band': round(upper_band, 4),
            'lower_band': round(lower_band, 4),
            'volume': volume,
            'avg_volume': round(avg_volume, 2),
            'atr': round(atr, 4),
            'avg_atr': round(avg_atr, 4),
            'spread_pct': round(spread_pct, 6)
        }
        
        if spread_pct > self.max_spread_pct:
            return Signal(action='HOLD', confidence=0.0, meta={**meta, 'reason': 'wide_spread'})
            
        if atr < avg_atr * self.min_atr_ratio:
            return Signal(action='HOLD', confidence=0.0, meta={**meta, 'reason': 'low_volatility'})
            
        high_volume = volume > avg_volume * self.volume_multiplier
        
        if price < lower_band and high_volume:
            deviation = (lower_band - price) / std_dev
            confidence = min(0.95, 0.55 + (deviation * 0.08))
            return Signal(action='BUY', confidence=round(confidence, 2), meta=meta)
        elif price > upper_band and high_volume:
            deviation = (price - upper_band) / std_dev
            confidence = min(0.95, 0.55 + (deviation * 0.08))
            return Signal(action='SELL', confidence=round(confidence, 2), meta=meta)
        else:
            return Signal(action='HOLD', confidence=0.0, meta=meta)