class BidAskMicrostructureStrategy(BaseStrategy):
    def __init__(self, momentum_window: int = 5, pressure_threshold: float = 0.6):
        super().__init__()
        self.momentum_window = momentum_window
        self.pressure_threshold = pressure_threshold
        self.price_changes = deque(maxlen=momentum_window)
        self.last_price = None
        
    def reset(self) -> None:
        self.price_changes.clear()
        self.last_price = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        timestamp = tick_data.timestamp
        
        if bid <= 0 or ask <= 0 or ask <= bid:
            return Signal(action='HOLD', confidence=0.0, meta={'timestamp': timestamp, 'error': 'invalid_spread'})
        
        # Calculate microstructure pressure: 1.0 = at ask (buying), -1.0 = at bid (selling)
        spread_position = (price - bid) / (ask - bid)
        pressure = (spread_position - 0.5) * 2  # Normalize to -1.0 to 1.0
        
        if self.last_price is None:
            self.last_price = price
            return Signal(action='HOLD', confidence=0.0, meta={'timestamp': timestamp, 'pressure': pressure})
        
        pct_change = (price - self.last_price) / self.last_price if self.last_price != 0 else 0
        self.price_changes.append(pct_change)
        self.last_price = price
        
        if len(self.price_changes) < self.momentum_window:
            return Signal(action='HOLD', confidence=0.0, 
                         meta={'timestamp': timestamp, 'pressure': pressure, 'status': 'building_momentum'})
        
        avg_momentum = sum(self.price_changes) / len(self.price_changes)
        
        # Buy signal: High buying pressure + positive momentum
        if pressure > self.pressure_threshold and avg_momentum > 0:
            conf = min(1.0, (pressure + min(abs(avg_momentum) * 100, 1.0)) / 2)
            return Signal(action='BUY