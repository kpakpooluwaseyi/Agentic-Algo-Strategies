class BidAskScalpingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.prev_price = None
        self.velocity_samples = deque(maxlen=3)
        
    def reset(self) -> None:
        self.prev_price = None
        self.velocity_samples.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        spread = tick_data.ask - tick_data.bid
        if spread <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        mid_price = (tick_data.bid + tick_data.ask) / 2
        relative_position = (tick_data.price - tick_data.bid) / spread
        
        if self.prev_price is not None:
            velocity = tick_data.price - self.prev_price
            self.velocity_samples.append(velocity)
        self.prev_price = tick_data.price
        
        avg_velocity = sum(self.velocity_samples) / len(self.velocity_samples) if self.velocity_samples else 0
        volatility_normalized = abs(avg_velocity) / (tick_data.atr + 1e-9) if tick_data.atr > 0 else 0