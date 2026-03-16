class BidAskImbalance scalper(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.volume_history = deque(maxlen=20)
        self.bid_touch_count = 0
        self.ask_touch_count = 0
        self.last_price = None
        self.last_signal_type = None
        
    def reset(self) -> None:
        self.volume_history.clear()
        self.bid_touch_count = 0
        self.ask_touch_count = 0
        self.last_price = None
        self.last_signal_type = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        if len(self.volume_history) > 0:
            avg_volume = sum(self.volume_history) / len(self.volume_history)
        else:
            avg_volume = volume
            
        tolerance = 0.0001
        
        if abs(price - bid) < tolerance:
            self.bid_touch_count += 1
            self.ask_touch_count = 0
        elif abs(price - ask) < tolerance:
            self.ask_touch_count += 1
            self.bid_touch_count = 0
        else:
            self.bid_touch_count = max(0, self.bid_touch_count - 1)
            self.ask_touch_count = max(0, self.ask_touch_count - 1)
            
        signal = None
        
        if self.ask_touch_count >= 3 and volume > avg_volume * 1.2:
            if self.last_signal_type != 'BUY':
                confidence = min(1.0, 0.5 + (self.ask_touch_count * 0.1) + ((volume / avg_volume) - 1) * 0.2)
                signal = Signal(
                    action='BUY', 
                    confidence=confidence, 
                    meta={'pressure': 'ask', 'consecutive_hits': self.ask_touch_count}
                )
                self.last_signal_type