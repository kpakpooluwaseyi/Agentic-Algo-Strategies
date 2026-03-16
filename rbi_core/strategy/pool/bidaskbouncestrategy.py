class BidAskBounceStrategy(BaseStrategy):
    def __init__(self, ma_window: int = 15, deviation_threshold: float = 1.8):
        self.ma_window = ma_window
        self.deviation_threshold = deviation_threshold
        self.midpoints = deque(maxlen=ma_window)
        self.cooldown = 0
        self.cooldown_period = 3
        
    def reset(self) -> None:
        self.midpoints.clear()
        self.cooldown = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        if self.cooldown > 0:
            self.cooldown -= 1
            
        midpoint = (tick_data.bid + tick_data.ask) / 2
        self.midpoints.append(midpoint)
        
        if len(self.midpoints) < self.ma_window or self.cooldown > 0:
            return None
            
        sma = sum(self.midpoints) / len(self.midpoints)
        current_price = tick_data.price
        current_atr = tick_data.atr
        
        if current_atr <= 0:
            return None
            
        distance = (current_price - sma) / current_atr
        
        if distance > self.deviation_threshold:
            confidence = min(abs(distance) / (self.deviation_threshold * 2), 1.0)
            self.cooldown = self.cooldown_period
            return Signal(action='SELL', confidence=confidence, meta={'distance_from_mean': distance, 'fair_value': sma, 'regime': 'overbought'})
        elif distance < -self.deviation_threshold:
            confidence = min(abs(distance) / (self.deviation_threshold * 2), 1.0)
            self.cooldown = self.cooldown_period
            return Signal(action='BUY', confidence=confidence, meta={'distance_from_mean': distance, 'fair_value': sma, 'regime': 'oversold'})
            
        return None