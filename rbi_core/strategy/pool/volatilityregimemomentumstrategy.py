class VolatilityRegimeMomentumStrategy(BaseStrategy):
    def __init__(self):
        self.prices = deque(maxlen=30)
        self.atrs = deque(maxlen=10)
        
    def reset(self) -> None:
        self.prices.clear()
        self.atrs.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        timestamp = tick_data.timestamp
        atr = tick_data.atr
        
        self.prices.append(price)
        self.atrs.append(atr)
        
        if len(self.prices) < 30 or len(self.atrs) < 10:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        old_price = self.prices[0]
        momentum = (price - old_price) / old_price if old_price > 0 else 0
        
        avg_atr = sum(self.atrs) / len(self.atrs)
        atr_increasing = atr > avg_atr
        
        short_ma = sum(list(self.prices)[-5:]) / 5
        long_ma = sum(self.prices) / len(self.prices)
        
        if short_ma > long_ma and momentum > 0 and atr_increasing:
            confidence = min(1.0, abs(momentum) * 100)
            return Signal(action='BUY', confidence=confidence, meta={'momentum': momentum, 'atr_ratio': atr/avg_atr})
        elif short_ma < long_ma and momentum < 0 and atr_increasing:
            confidence = min(1.0, abs(momentum) * 100)
            return Signal(action='SELL', confidence=confidence, meta={'momentum': momentum, 'atr_ratio': atr/avg_atr})
        else:
            return Signal(action='HOLD', confidence=0.0, meta={})