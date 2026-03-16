class BidAskImbalance(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.lookback = 10
        self.pressure_threshold = 0.75
        self.reset()
    
    def reset(self) -> None:
        self.position_in_spread_history = deque(maxlen=self.lookback)
        self.volume_history = deque(maxlen=self.lookback)
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        if ask <= bid:
            return Signal(action='HOLD', confidence=0.0, meta={'spread': ask - bid})
        
        pos_in_spread = (price - bid) / (ask - bid)
        
        self.position_in_spread_history.append(pos_in_spread)
        self.volume_history.append(volume)
        
        if len(self.position_in_spread_history) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'pressure': pos_in_spread})
        
        avg_volume = sum(self.volume_history) / self.lookback
        avg_position = sum(self.position_in_spread_history) / self.lookback
        
        volume_surge = volume / (avg_volume + 1e-9) if avg_volume > 0 else 1.0
        
        action = 'HOLD'
        confidence = 0.0
        
        if pos_in_spread > self.pressure_threshold and volume_surge > 1.5:
            action = 'BUY'
            confidence = min(0.85, pos_in_spread * 0.5 + (volume_surge - 1) * 0.3)
        elif pos_in_spread < (1 - self.pressure_threshold) and volume_surge > 1.5:
            action = 'SELL'
            confidence = min(0.85, (1 - pos_in_spread) * 0.5 + (volume_surge - 1) * 0.3)
        
        return Signal(action=action, confidence=confidence, meta={
            'position_in_spread': pos_in_spread,
            'avg_position': avg_position,
            'volume_surge': volume_surge,
            'spread': ask - bid
        })