class VolumeImpulseStrategy(BaseStrategy):
    def __init__(self, volume_lookback: int = 20, price_lookback: int = 5, threshold: float = 2.0):
        self.volume_lookback = volume_lookback
        self.price_lookback = price_lookback
        self.threshold = threshold
        self.volumes = deque(maxlen=volume_lookback)
        self.prices = deque(maxlen=price_lookback)
        
    def reset(self) -> None:
        self.volumes.clear()
        self.prices.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.volumes) < self.volume_lookback or len(self.prices) < 2:
            return None
            
        avg_volume = statistics.mean(self.volumes)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio > self.threshold:
            price_change = (self.prices[-1] - self.prices[0]) / self.prices[0] if self.prices[0] != 0 else 0
            
            if price_change > 0:
                confidence = min(1.0, (volume_ratio - self.threshold) / self.threshold)
                return Signal(
                    action='BUY',
                    confidence=confidence,
                    meta={'volume_ratio': volume_ratio, 'price_change_pct': price_change}
                )
            elif price_change < 0:
                confidence = min(1.0, (volume_ratio - self.threshold) / self.threshold)
                return Signal(
                    action='SELL',
                    confidence=confidence,
                    meta={'volume_ratio': volume_ratio, 'price_change_pct': price_change}
                )
                
        return Signal(action='HOLD', confidence=0.0, meta={})