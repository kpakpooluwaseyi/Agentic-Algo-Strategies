class VolumeSpreadStrategy(BaseStrategy):
    def __init__(self, volume_lookback: int = 10, spread_threshold: float = 0.3, volume_threshold: float = 1.5):
        self.volume_lookback = volume_lookback
        self.spread_threshold = spread_threshold
        self.volume_threshold = volume_threshold
        self.volumes: List[float] = []
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        self.volumes.append(volume)
        if len(self.volumes) > self.volume_lookback:
            self.volumes.pop(0)
            
        if len(self.volumes) < self.volume_lookback:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        avg_volume = sum(self.volumes) / len(self.volumes)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        spread = ask - bid
        if spread <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={'error': 'invalid_spread'})
            
        mid = (bid + ask) / 2.0
        position = (price - mid) / (spread / 2.0) if spread > 0 else 0.0
        position = max(-1.0, min(1.0, position))
        
        meta = {
            'mid': mid,
            'spread': spread,
            'position': position,
            'volume_ratio': volume_ratio
        }
        
        if position > (1 - self.spread_threshold) and volume_ratio > self.volume_threshold:
            confidence = min((position - (1 - self.spread_threshold)) / self.spread_threshold, 1.0)
            confidence *= min(volume_ratio / self.volume_threshold, 1.0)
            return Signal(action='BUY', confidence=confidence, meta=meta)
            
        elif position < -(1 - self.spread_threshold) and volume_ratio > self.volume_threshold:
            confidence = min((abs(position) - (1 - self.spread_threshold)) / self.spread_threshold, 1.0)
            confidence *= min(volume_ratio / self.volume_threshold, 1.0)
            return Signal(action='SELL', confidence=confidence, meta=meta)
            
        return Signal(action='HOLD', confidence=0.0, meta=meta)
        
    def reset(self) -> None:
        self.volumes.clear()