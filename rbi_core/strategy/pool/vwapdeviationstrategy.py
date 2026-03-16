class VWAPDeviationStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.window = deque(maxlen=30)
        
    def reset(self) -> None:
        self.window.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        if volume <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        self.window.append((price, volume))
        
        if len(self.window) < 30:
            return Signal(action='HOLD', confidence=0.0, meta={'accumulating': len(self.window)})
        
        total_pv = sum(p * v for p, v in self.window)
        total_v = sum(v for p, v in self.window)
        vwap = total_pv / total_v
        
        deviation = (price - vwap) / vwap if vwap != 0 else 0
        
        if deviation < -0.015:
            confidence = min(abs(deviation) * 20, 1.0)
            return Signal(
                action='BUY', 
                confidence=confidence, 
                meta={'indicator': 'vwap_deviation', 'vwap': vwap, 'deviation': deviation}
            )
        elif deviation > 0.015:
            confidence = min(deviation * 20, 1.0)
            return Signal(
                action='SELL', 
                confidence=confidence, 
                meta={'indicator': 'vwap_deviation', 'vwap': vwap, 'deviation': deviation}
            )
        else:
            return Signal(
                action='HOLD', 
                confidence=0.0, 
                meta={'indicator': 'vwap_deviation', 'vwap': vwap, 'deviation': deviation}
            )