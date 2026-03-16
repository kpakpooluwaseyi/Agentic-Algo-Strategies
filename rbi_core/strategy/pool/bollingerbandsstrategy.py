class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__()
        self.window = window
        self.num_std = num_std
        self.prices = deque(maxlen=window)
        self.position = 0
    
    def reset(self) -> None:
        self.prices.clear()
        self.position = 0
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        self.prices.append(price)
        
        if len(self.prices) < self.window:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        mean = sum(self.prices) / self.window
        variance = sum((p - mean) ** 2 for p in self.prices) / self.window
        std = math.sqrt(variance)
        
        upper = mean + self.num_std * std
        lower = mean - self.num_std * std
        
        action = 'HOLD'
        confidence = 0.0
        
        if price < lower and self.position != 1:
            action = 'BUY'
            confidence = min(1.0, (lower - price) / (std + 1e-9))
            self.position = 1
        elif price > upper and self.position != -1:
            action = 'SELL'
            confidence = min(1.0, (price - upper) / (std + 1e-9))
            self.position = -1
        
        return Signal(action=action, confidence=confidence, meta={'mean': mean, 'upper': upper, 'lower': lower})