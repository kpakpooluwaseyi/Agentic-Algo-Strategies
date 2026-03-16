class VolumeConfirmedMomentumStrategy(BaseStrategy):
    def __init__(self, window: int = 10, volume_threshold: float = 1.5):
        self.window = window
        self.volume_threshold = volume_threshold
        self.volumes = deque(maxlen=window)
        self.prev_price = None
    
    def reset(self) -> None:
        self.volumes.clear()
        self.prev_price = None
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        signal = None
        
        if len(self.volumes) == self.window and self.prev_price is not None and self.prev_price != 0:
            avg_volume = sum(self.volumes) / self.window
            price_change = (price - self.prev_price) / self.prev_price
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio > self.volume_threshold:
                if price_change > 0:
                    conf = min(0.6 + (volume_ratio - self.volume_threshold) * 0.25, 1.0)
                    signal = Signal(action='BUY', confidence=conf, meta={'volume_ratio': volume_ratio, 'return': price_change})
                elif price_change < 0:
                    conf = min(0.6 + (volume_ratio - self.volume_threshold) * 0.25, 1.0)
                    signal = Signal(action='SELL', confidence=conf, meta={'volume_ratio': volume_ratio, 'return': price_change})
                else:
                    signal = Signal(action='HOLD', confidence=0.0, meta={})
            else:
                signal = Signal(action='HOLD', confidence=0.0, meta={})
        
        self.prev_price = price
        self.volumes.append(volume)
        return signal