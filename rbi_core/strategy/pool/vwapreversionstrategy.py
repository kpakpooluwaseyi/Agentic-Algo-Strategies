class VWAPReversionStrategy(BaseStrategy):
    def __init__(self, lookback: int = 25, deviation_threshold: float = 0.015):
        super().__init__()
        self.lookback = lookback
        self.deviation_threshold = deviation_threshold
        self.data: Deque[Tuple[float, float]] = deque(maxlen=lookback)
        
    def reset(self) -> None:
        self.data.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        self.data.append((price, volume))
        
        if len(self.data) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'buffer': len(self.data)})
        
        total_pv = sum(p * v for p, v in self.data)
        total_v = sum(v for _, v in self.data)
        vwap = total_pv / total_v if total_v > 0 else price
        
        deviation = (price - vwap) / vwap if vwap != 0 else 0
        
        action = 'HOLD'
        confidence = 0.0
        
        if deviation > self.deviation_threshold:
            action = 'SELL'
            confidence = min(1.0, abs(deviation) / (self.deviation_threshold * 2))
        elif deviation < -self.deviation_threshold:
            action = 'BUY'
            confidence = min(1.0, abs(deviation) / (self.deviation_threshold * 2))
            
        return Signal(action=action, confidence=confidence, meta={
            'vwap': vwap,
            'deviation_pct': deviation * 100,
            'cumulative_volume': total_v
        })