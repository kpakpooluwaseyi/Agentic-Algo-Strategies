class ZScoreMeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback: int = 30, entry_z: float = 2.0, exit_z: float = 0.5):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.prices = deque(maxlen=lookback)
        self.position = 0
        self.reset()
    
    def reset(self) -> None:
        self.prices.clear()
        self.position = 0
    
    def _calculate_std(self, mean: float) -> float:
        if len(self.prices) < 2:
            return 0.0
        variance = sum((p - mean) ** 2 for p in self.prices) / (len(self.prices) - 1)
        return math.sqrt(variance)
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        timestamp = tick_data.timestamp
        
        self.prices.append(price)
        
        if len(self.prices) < self.lookback:
            return None
        
        mean = sum(self.prices) / len(self.prices)
        std = self._calculate_std(mean)
        
        if std == 0:
            return None
        
        z_score = (price - mean) / std
        
        action = 'HOLD'
        confidence = 0.0
        
        if z_score < -self.entry_z and self.position == 0:
            action = 'BUY'
            confidence = min(1.0, abs(z_score) / (self.entry_z * 2))
            self.position = 1
        elif z_score > self.entry_z and self.position == 0:
            action = 'SELL'
            confidence = min(1.0, abs(z_score) / (self.entry_z * 2))
            self.position = -1
        elif abs(z_score) < self.exit_z and self.position != 0:
            action = 'SELL' if self.position == 1 else 'BUY'
            confidence = 0.5
            self.position = 0
        
        if action == 'HOLD':
            return None
        
        return Signal(
            action=action,
            confidence=confidence,
            meta={
                'z_score': z_score,
                'mean': mean,
                'std': std,
                'timestamp': timestamp
            }
        )