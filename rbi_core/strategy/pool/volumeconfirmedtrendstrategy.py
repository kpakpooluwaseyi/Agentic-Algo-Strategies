class VolumeConfirmedTrendStrategy(BaseStrategy):
    def __init__(self, window: int = 10, volume_threshold: float = 1.5, trend_threshold: float = 0.5):
        super().__init__()
        self.window = window
        self.volume_threshold = volume_threshold
        self.trend_threshold = trend_threshold
        self.price_history: Deque[float] = deque(maxlen=window)
        self.volume_history: Deque[float] = deque(maxlen=window)
        
    def reset(self) -> None:
        super().reset()
        self.price_history.clear()
        self.volume_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        if len(self.price_history) < self.window:
            self.price_history.append(price)
            self.volume_history.append(volume)
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'initializing'})
            
        start_price = self.price_history[0]
        price_change_pct = (price - start_price) / (start_price + 1e-9)
        
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        volume_surge = volume / (avg_volume + 1e-9)
        
        volatility_normalized = price_change_pct / ((atr / price) + 1e-9) if atr > 0 else 0.0
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if volatility_normalized > self.trend_threshold and volume_surge > self.volume_threshold:
            confidence = min(1.0, (volatility_normalized / (self.trend_threshold * 2)) * 0.6 + (volume_surge / (self.volume_threshold * 2)) * 0.4)
            return Signal(action='BUY', confidence=confidence, meta={'trend': volatility_normalized, 'volume_surge': volume_surge})
        elif volatility_normalized < -self.trend_threshold and volume_surge > self.volume_threshold:
            confidence = min(1.0, (abs(volatility_normalized) / (self.trend_threshold * 2)) * 0.6 + (volume_surge / (self.volume_threshold * 2)) * 0.4)
            return Signal(action='SELL', confidence=confidence, meta={'trend': volatility_normalized, 'volume_surge': volume_surge})
        else:
            return Signal(action='HOLD', confidence=0.0, meta={'trend': volatility_normalized, 'volume_ratio': volume_surge})