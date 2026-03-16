class MicrostructurePressureStrategy(BaseStrategy):
    def __init__(self, volume_window: int = 10, pressure_threshold: float = 0.7):
        self.volume_window = volume_window
        self.pressure_threshold = pressure_threshold
        self.volumes: List[float] = []
        
    def reset(self) -> None:
        self.volumes = []
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        spread = ask - bid
        if spread <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        micro_pressure = (price - bid) / spread
        
        self.volumes.append(volume)
        if len(self.volumes) > self.volume_window:
            self.volumes.pop(0)
            
        if len(self.volumes) < self.volume_window:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        avg_volume = sum(self.volumes) / len(self.volumes)
        
        if volume > avg_volume:
            if micro_pressure > self.pressure_threshold:
                confidence = min(1.0, (micro_pressure - 0.5) * 2)
                return Signal(action='BUY', confidence=confidence, meta={'pressure': micro_pressure, 'spread': spread})
            elif micro_pressure < (1 - self.pressure_threshold):
                confidence = min(1.0, (0.5 - micro_pressure) * 2)
                return Signal(action='SELL', confidence=confidence, meta={'pressure': micro_pressure, 'spread': spread})
                
        return Signal(action='HOLD', confidence=0.0, meta={})