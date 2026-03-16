class ConsecutiveMomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.prev_price: Optional[float] = None
        self.prev_volume: Optional[float] = None
        self.up_streak = 0
        self.down_streak = 0
        self.current_direction: Optional[str] = None
        
    def reset(self) -> None:
        self.prev_price = None
        self.prev_volume = None
        self.up_streak = 0
        self.down_streak = 0
        self.current_direction = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        if self.prev_price is None:
            self.prev_price = price
            self.prev_volume = volume
            return Signal(action='HOLD', confidence=0.0, meta={'init': True})
            
        tick_return = (price - self.prev_price) / self.prev_price if self.prev_price != 0 else 0
        volume_delta = volume - self.prev_volume if self.prev_volume is not None else 0
        
        spread_pct = (ask - bid) / price if price > 0 else 0
        
        if tick_return > 0:
            self.up_streak += 1
            self.down_streak = 0
        elif tick_return < 0:
            self.down_streak += 1
            self.up_streak = 0
        else:
            self.up_streak = max(0, self.up_streak - 1)
            self.down_streak = max(0, self.down_streak - 1)
            
        signal = Signal(action='HOLD', confidence=0.0, meta={})
        
        if self.up_streak >= 4 and volume_delta > 0 and self.current_direction != 'LONG':
            self.current_direction = 'LONG'
            confidence = min(0.88, 0.55 + self.up_streak * 0.03 - spread_pct * 10)
            signal = Signal(
                action='BUY',
                confidence=max(0.5, confidence),
                meta={'streak': self.up_streak, 'return_pct': round(tick_return * 100, 4), 'volume_delta': volume_delta}
            )
        elif self.down_streak >= 4 and volume_delta > 0 and self.current_direction != 'SHORT