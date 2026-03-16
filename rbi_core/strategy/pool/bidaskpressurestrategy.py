class BidAskPressureStrategy(BaseStrategy):
    def __init__(self, volume_lookback: int = 10, spread_threshold: float = 0.05, pressure_threshold: float = 0.7):
        self.volume_lookback = volume_lookback
        self.spread_threshold = spread_threshold
        self.pressure_threshold = pressure_threshold
        self.volume_history: Deque[float] = deque(maxlen=volume_lookback)
        
    def reset(self) -> None:
        self.volume_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        spread = ask - bid
        mid = (ask + bid) / 2
        
        if spread <= 0 or price <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={'error': 'invalid_prices'})
        
        spread_pct = spread / mid
        if spread_pct > self.spread_threshold:
            return Signal(action='HOLD', confidence=0.0, meta={'spread_too_wide': spread_pct})
        
        position = (price - bid) / spread
        avg_volume = sum(self.volume_history) / len(self.volume_history) if self.volume_history else volume
        volume_surge = volume / (avg_volume + 1e-10) if avg_volume > 0 else 1.0
        
        action = 'HOLD'
        confidence = 0.0
        
        if position > self.pressure_threshold and volume_surge > 1.2:
            action = 'BUY'
            confidence = min(1.0, position * (volume_surge - 1.0))
        elif position < (1 - self.pressure_threshold) and volume_surge > 1.2:
            action = 'SELL'
            confidence = min(1.0, (1 - position) * (volume_surge - 1.0))
        
        self.volume_history.append(volume)
        
        return Signal(
            action=action,
            confidence=confidence,
            meta={'position_in_spread': position, 'volume_surge': volume_surge, 'spread': spread}
        )