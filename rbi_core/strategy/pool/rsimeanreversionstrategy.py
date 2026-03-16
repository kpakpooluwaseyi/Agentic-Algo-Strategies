class RSIMeanReversionStrategy(BaseStrategy):
    def __init__(self):
        self.price_changes = deque(maxlen=14)
        self.prev_price = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        
        if self.prev_price is None:
            self.prev_price = price
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        change = price - self.prev_price
        self.prev_price = price
        self.price_changes.append(change)
        
        if len(self.price_changes) < 14:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        gains = [c for c in self.price_changes if c > 0]
        losses = [-c for c in self.price_changes if c < 0]
        
        avg_gain = sum(gains) / 14 if gains else 0.0
        avg_loss = sum(losses) / 14 if losses else 0.0
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        if rsi < 30:
            confidence = (30 - rsi) / 30
            return Signal(action='BUY', confidence=confidence, meta={'rsi': rsi, 'avg_gain': avg_gain})
        elif rsi > 70:
            confidence = (rsi - 70) / 30
            return Signal(action='SELL', confidence=confidence, meta={'rsi': rsi, 'avg_loss': avg_loss})
        else:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
    def reset(self) -> None:
        self.price_changes.clear()
        self.prev_price = None