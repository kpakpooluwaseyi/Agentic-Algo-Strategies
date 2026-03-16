class BidAskVolumeImbalanceStrategy(BaseStrategy):
    def __init__(self, volume_window: int = 10, surge_threshold: float = 1.5):
        super().__init__()
        self.volume_window = volume_window
        self.surge_threshold = surge_threshold
        self.volume_history: Deque[float] = deque(maxlen=volume_window)
        self.prev_bid: Optional[float] = None
        self.prev_ask: Optional[float] = None
        
    def reset(self) -> None:
        self.volume_history.clear()
        self.prev_bid = None
        self.prev_ask = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        current_volume = tick_data.volume
        current_bid = tick_data.bid
        current_ask = tick_data.ask
        
        if len(self.volume_history) < self.volume_window:
            self.volume_history.append(current_volume)
            self.prev_bid = current_bid
            self.prev_ask = current_ask
            return None
            
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        
        bid_change = current_bid - self.prev_bid if self.prev_bid else 0.0
        ask_change = current_ask - self.prev_ask if self.prev_ask else 0.0
        
        action = 'HOLD'
        confidence = 0.0
        
        if current_volume > avg_volume * self.surge_threshold:
            if bid_change > 0 and ask_change >= 0:
                action = 'BUY'
                volume_ratio = current_volume / avg_volume
                confidence = min(0.9, 0.4 + (volume_ratio - self.surge_threshold) * 0.3)
            elif ask_change < 0 and bid_change <= 0:
                action = 'SELL'
                volume_ratio = current_volume / avg_volume
                confidence = min(0.9, 0.4 + (volume_ratio - self.surge_threshold) * 0.3)
                
        self.volume_history.append(current_volume)
        self.prev_bid = current_bid
        self.prev_ask = current_ask
        
        return Signal(action=action, confidence=confidence, meta={
            'avg_volume': avg_volume,
            'bid_change': bid_change,
            'ask_change': ask_change,
            'volume_surge': current_volume > avg_volume * self.surge_threshold
        })