class VolumeWeightedMACStrategy(BaseStrategy):
    def __init__(self, fast_period: int = 5, slow_period: int = 15):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.pv_values = deque(maxlen=slow_period)
        self.volumes = deque(maxlen=slow_period)
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        self.pv_values.append(price * volume)
        self.volumes.append(volume)
        
        if len(self.pv_values) < self.slow_period:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        fast_pv = sum(list(self.pv_values)[-self.fast_period:])
        fast_vol = sum(list(self.volumes)[-self.fast_period:])
        slow_pv = sum(self.pv_values)
        slow_vol = sum(self.volumes)
        
        if fast_vol == 0 or slow_vol == 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        fast_vwma = fast_pv / fast_vol
        slow_vwma = slow_pv / slow_vol
        
        if fast_vwma > slow_vwma * 1.001:
            confidence = min(1.0, (fast_vwma / slow_vwma - 1) * 1000 + 0.3)
            return Signal(action='BUY', confidence=confidence, meta={'fast_vwma': fast_vwma, 'slow_vwma': slow_vwma})
        elif fast_vwma < slow_vwma * 0.999:
            confidence = min(1.0, (1 - fast_vwma / slow_vwma) * 1000 + 0.3)
            return Signal(action='SELL', confidence=confidence, meta={'fast_vwma': fast_vwma, 'slow_vwma': slow_vwma})
        return Signal(action='HOLD', confidence=0.0, meta={})
    
    def reset(self) -> None:
        self.pv_values.clear()
        self.volumes.clear()