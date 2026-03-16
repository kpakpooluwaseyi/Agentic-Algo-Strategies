class VolumeMomentumStrategy(BaseStrategy):
    def __init__(self, lookback: int = 10, threshold: float = 0.05):
        super().__init__()
        self.lookback = lookback
        self.threshold = threshold
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        timestamp = tick_data.timestamp
        
        if len(self.prices) > 0:
            prev_price = self.prices[-1]
            price_change = (price - prev_price) / prev_price if prev_price != 0 else 0
        else:
            price_change = 0
            
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'initializing', 'timestamp': timestamp})
            
        avg_volume = sum(self.volumes) / len(self.volumes)
        if avg_volume == 0:
            return Signal(action='HOLD', confidence=0.0, meta={'error': 'zero_average_volume', 'timestamp': timestamp})
            
        volume_ratio = volume / avg_volume
        momentum = price_change * volume_ratio
        
        meta = {
            'momentum': momentum,
            'price_change': price_change,
            'volume_ratio': volume_ratio,
            'timestamp': timestamp
        }
        
        if momentum > self.threshold:
            confidence = min(momentum / (self.threshold * 2), 1.0)
            return Signal(action='BUY', confidence=confidence, meta=meta)
        elif momentum < -self.threshold:
            confidence = min(abs(momentum) / (self.threshold * 2), 1.0)
            return Signal(action='SELL', confidence=confidence, meta=meta)
        else:
            return Signal(action='HOLD', confidence=0.0, meta=meta)