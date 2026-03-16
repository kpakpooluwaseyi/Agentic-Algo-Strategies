class CumulativeVWAPStrategy(BaseStrategy):
    def __init__(self, deviation_pct: float = 0.001):
        super().__init__()
        self.deviation_pct = deviation_pct
        self.cumulative_pv = 0.0
        self.cumulative_volume = 0.0
        
    def reset(self) -> None:
        self.cumulative_pv = 0.0
        self.cumulative_volume = 0.0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        self.cumulative_pv += price * volume
        self.cumulative_volume += volume
        
        if self.cumulative_volume > 0:
            vwap = self.cumulative_pv / self.cumulative_volume
            deviation = (price - vwap) / vwap if vwap != 0 else 0
            
            if deviation < -self.deviation_pct:
                confidence = min(0.95, 0.5 + abs(deviation) * 200)
                return Signal(action='BUY', confidence=confidence,
                            meta={'vwap': vwap, 'deviation_pct': deviation * 100})
            elif deviation > self.deviation_pct:
                confidence = min(0.95, 0.5 + deviation * 200)
                return Signal(action='SELL', confidence=confidence,
                            meta={'vwap': vwap, 'deviation_pct': deviation * 100})
        
        return Signal(action='HOLD', confidence=0.0, meta={})