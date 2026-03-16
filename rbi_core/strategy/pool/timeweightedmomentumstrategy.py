class TimeWeightedMomentumStrategy(BaseStrategy):
    def __init__(self, window: int = 10, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold
        self.prices: Deque[float] = deque(maxlen=window)
        self.timestamps: Deque[float] = deque(maxlen=window)
        self.returns: Deque[float] = deque(maxlen=window)
        
    def reset(self) -> None:
        self.prices.clear()
        self.timestamps.clear()
        self.returns.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        timestamp = tick_data.timestamp
        
        if len(self.prices) < 2:
            self.prices.append(price)
            self.timestamps.append(timestamp)
            return Signal(action='HOLD', confidence=0.0, meta={'initializing': True})
        
        dt = timestamp - self.timestamps[-1]
        if dt <= 0:
            dt = 1
        
        price_change = price - self.prices[-1]
        velocity = price_change / dt
        
        if len(self.returns) < self.window - 1:
            self.returns.append(velocity)
            self.prices.append(price)
            self.timestamps.append(timestamp)
            return Signal(action='HOLD', confidence=0.0, meta={'building_window': len(self.returns)})
        
        mean_velocity = sum(self.returns) / len(self.returns)
        variance = sum((v - mean_velocity) ** 2 for v in self.returns) / len(self.returns)
        std_velocity = math.sqrt(variance) if variance > 0 else 1e-10
        
        z_score = (velocity - mean_velocity) / std_velocity
        
        action = 'HOLD'
        confidence = 0.0
        
        if z_score > self.z_threshold and velocity > 0:
            action = 'BUY'
            confidence = min(1.0, z_score / 3.0)
        elif z_score < -self.z_threshold and velocity < 0:
            action = 'SELL'
            confidence = min(1.0, abs(z_score) / 3.0)
        
        self.returns.append(velocity)
        self.prices.append(price)
        self.timestamps.append(timestamp)
        
        return Signal(
            action=action,
            confidence=confidence,
            meta={'velocity': velocity, 'z_score': z_score, 'mean_velocity': mean_velocity}
        )