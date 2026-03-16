class MeanReversionSpreadStrategy(BaseStrategy):
    def __init__(self, window: int = 15, z_threshold: float = 2.0, max_spread_bps: float = 5.0):
        super().__init__()
        self.window = window
        self.z_threshold = z_threshold
        self.max_spread_bps = max_spread_bps
        self.prices: Deque[float] = deque(maxlen=window)
        self.current_position: int = 0
        
    def reset(self) -> None:
        self.prices.clear()
        self.current_position = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        
        spread_bps = ((ask - bid) / price) * 10000 if price > 0 else 0
        self.prices.append(price)
        
        if len(self.prices) < self.window:
            return None
            
        mean_price = sum(self.prices) / self.window
        variance = sum((p - mean_price) ** 2 for p in self.prices) / self.window
        std_dev = variance ** 0.5
        
        z_score = (price - mean_price) / std_dev if std_dev > 0 else 0
        
        action = 'HOLD'
        confidence = 0.0
        
        if spread_bps <= self.max_spread_bps:
            if z_score > self.z_threshold and self.current_position >= 0:
                action = 'SELL'
                confidence = min(abs(z_score) / 4.0, 1.0)
                self.current_position = -1
            elif z_score < -self.z_threshold and self.current_position <= 0:
                action = 'BUY'
                confidence = min(abs(z_score) / 4.0, 1.0)
                self.current_position = 1
        
        if action == 'HOLD':
            return None
            
        return Signal(
            action=action,
            confidence=confidence,
            meta={
                'z_score': z_score,
                'mean_price': mean_price,
                'std_dev': std_dev,
                'spread_bps': spread_bps,
                'deviation_from_mean': price - mean_price
            }
        )