class BidAskImbalanceStrategy(BaseStrategy):
    def __init__(self, lookback: int = 10, sentiment_threshold: float = 0.65, volume_acceleration_factor: float = 1.3):
        super().__init__()
        self.lookback = lookback
        self.sentiment_threshold = sentiment_threshold
        self.volume_acceleration_factor = volume_acceleration_factor
        self.volumes = deque(maxlen=lookback)
        self.price_changes = deque(maxlen=lookback)
        self.last_price = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        if self.last_price is not None:
            self.price_changes.append(tick_data.price - self.last_price)
        self.volumes.append(tick_data.volume)
        self.last_price = tick_data.price
        
        if len(self.volumes) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        spread = tick_data.ask - tick_data.bid
        if spread <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        relative_position = (tick_data.price - tick_data.bid) / spread
        
        avg_volume = sum(self.volumes) / len(self.volumes)
        volume_surge = tick_data.volume > avg_volume * self.volume_acceleration_factor
        
        recent_momentum = sum(self.price_changes) / len(self.price_changes) if self.price_changes else 0
        
        meta = {
            'relative_position': relative_position,
            'volume_surge': volume_surge,
            'spread': spread
        }
        
        if relative_position > self.sentiment_threshold and volume_surge and recent_momentum >= 0:
            confidence = min(0.9, relative_position * 0.8 + 0.2)
            return Signal(action='BUY', confidence=confidence, meta={**meta, 'reason': 'ask_pressure_with_volume'})
            
        elif relative_position < (1 - self.sentiment_threshold) and volume_surge and recent_momentum <= 0:
            confidence = min(0.9, (1 - relative_position) * 0.8 + 0.2)
            return Signal(action='SELL', confidence=confidence, meta={**meta, 'reason': 'bid_pressure_with_volume'})
            
        return Signal(action='HOLD', confidence=0.0, meta=meta)
        
    def reset(self) -> None:
        self.volumes.clear()
        self.price_changes.clear()
        self.last_price = None