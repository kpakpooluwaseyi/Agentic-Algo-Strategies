class RollingVWAPMeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback: int = 50, deviation_threshold: float = 2.0):
        self.lookback = lookback
        self.deviation_threshold = deviation_threshold
        self.data = deque(maxlen=lookback)
        self.reset()
    
    def reset(self) -> None:
        self.data.clear()
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        self.data.append((price, volume))
        
        if len(self.data) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'samples': len(self.data)})
        
        total_pv = sum(p * v for p, v in self.data)
        total_v = sum(v for _, v in self.data)
        vwap = total_pv / (total_v + 1e-9)
        
        distance = abs(price - vwap)
        threshold = atr * self.deviation_threshold
        
        if distance > threshold:
            confidence = min(1.0, (distance - threshold) / (atr + 1e-9))
            if price > vwap:
                return Signal(action='SELL', confidence=confidence,
                            meta={'vwap': vwap, 'distance': distance, 'atr': atr, 'bias': 'overbought'})
            else:
                return Signal(action='BUY', confidence=confidence,
                            meta={'vwap': vwap, 'distance': distance, 'atr': atr, 'bias': 'oversold'})
        
        return Signal(action='HOLD', confidence=0.0, meta={'vwap': vwap, 'distance': distance})