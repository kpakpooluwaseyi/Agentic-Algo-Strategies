class BollingerMeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback: int = 15, z_threshold: float = 2.0):
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.prices: Deque[float] = deque(maxlen=lookback)
        self.reset()
    
    def reset(self) -> None:
        self.prices.clear()
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        
        if len(self.prices) < self.lookback:
            self.prices.append(price)
            return None
        
        mean = sum(self.prices) / len(self.prices)
        variance = sum((p - mean) ** 2 for p in self.prices) / len(self.prices)
        std_dev = math.sqrt(variance + 1e-9)
        
        z_score = (price - mean) / std_dev
        
        spread = ask - bid
        mid_price = (bid + ask) / 2
        position_in_spread = (price - mid_price) / (spread + 1e-9) if spread > 0 else 0
        
        action = 'HOLD'
        confidence = 0.0
        
        if z_score > self.z_threshold:
            action = 'SELL'
            confidence = min(abs(z_score) / 4, 1.0)
        elif z_score < -self.z_threshold:
            action = 'BUY'
            confidence = min(abs(z_score) / 4, 1.0)
        
        self.prices.append(price)
        
        if action != 'HOLD':
            return Signal(action=action, confidence=confidence, meta={
                'z_score': z_score,
                'mean': mean,
                'std_dev': std_dev,
                'spread': spread,
                'position_in_spread': position_in_spread
            })
        return None