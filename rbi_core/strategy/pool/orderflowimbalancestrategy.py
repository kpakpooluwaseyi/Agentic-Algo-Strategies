class OrderFlowImbalanceStrategy(BaseStrategy):
    def __init__(self, volume_threshold: float = 1.5, pressure_threshold: float = 0.0001):
        self.volume_threshold = volume_threshold
        self.pressure_threshold = pressure_threshold
        self.volume_history = deque(maxlen=20)
        self.prev_bid = None
        self.prev_ask = None
        self.prev_mid = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        if len(self.volume_history) < 20:
            self.volume_history.append(volume)
            self.prev_bid = bid
            self.prev_ask = ask
            self.prev_mid = (bid + ask) / 2
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        current_mid = (bid + ask) / 2
        
        bid_change = bid - self.prev_bid
        ask_change = ask - self.prev_ask
        mid_change = current_mid - self.prev_mid
        
        volume_spike = volume > (avg_volume * self.volume_threshold)
        
        if volume_spike:
            if bid_change > self.pressure_threshold and mid_change > 0 and price >= current_mid:
                confidence = min((volume / avg_volume - 1) / self.volume_threshold, 1.0)
                signal = Signal(action='BUY', confidence=confidence, meta={'pressure': 'bid_lift', 'volume_ratio': volume/avg_volume})
            elif ask_change < -self.pressure_threshold and mid_change < 0 and price <= current_mid:
                confidence = min((volume / avg_volume - 1) / self.volume_threshold, 1.0)
                signal = Signal(action='SELL', confidence=confidence, meta={'pressure': 'ask_hit', 'volume_ratio': volume/avg_volume})
            else:
                signal = Signal(action='HOLD', confidence=0.0, meta={'flow': 'neutral'})
        else:
            signal = Signal(action='HOLD', confidence=0.0, meta={'flow': 'low_volume'})
        
        self.volume_history.append(volume)
        self.prev_bid = bid
        self.prev_ask = ask
        self.prev_mid = current_mid
        return signal
    
    def reset(self) -> None:
        self.volume_history.clear()
        self.prev_bid = None
        self.prev_ask = None
        self.prev_mid = None