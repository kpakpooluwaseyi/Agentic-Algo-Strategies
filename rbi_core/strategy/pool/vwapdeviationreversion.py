class VWAPDeviationReversion(BaseStrategy):
    def __init__(self, window: int = 15, deviation_threshold: float = 2.0):
        super().__init__()
        self.window = window
        self.deviation_threshold = deviation_threshold
        self.prices = deque(maxlen=window)
        self.volumes = deque(maxlen=window)
        self.last_signal_tick = 0
        self.tick_count = 0
        
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.last_signal_tick = 0
        self.tick_count = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        bid = tick_data.bid
        ask = tick_data.ask
        
        self.tick_count += 1
        
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) < self.window:
            return Signal(action='HOLD', confidence=0.0, meta={'buffering': len(self.prices)})
            
        total_vol = sum(self.volumes)
        if total_vol == 0:
            return Signal(action='HOLD', confidence=0.0, meta={'error': 'zero_volume'})
            
        vwap = sum(p * v for p, v in zip(self.prices, self.volumes)) / total_vol
        
        if atr <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={'atr': 'invalid'})
            
        deviation = (price - vwap) / atr
        
        spread = ask - bid
        spread_pct = spread / price if price > 0 else 0
        
        meta = {
            'vwap': vwap,
            'deviation': deviation,
            'spread_pct': spread_pct,
            'timestamp': tick_data.timestamp
        }
        
        cooldown_active = (self.tick_count - self.last_signal_tick) < 10
        
        if not cooldown_active:
            if deviation > self.deviation_threshold:
                self.last_signal_tick = self.tick_count
                confidence = min(deviation / (self.deviation_threshold * 2), 0.9)
                return Signal(action='SELL', confidence=confidence, meta={**meta, 'reason': 'overextended_above_vwap'})
                
            if deviation < -self.deviation_threshold:
                self.last_signal_tick = self.tick_count
                confidence = min(abs(deviation) / (self.deviation_threshold * 2), 0.9)
                return Signal(action='BUY', confidence=confidence, meta={**meta, 'reason': 'overextended_below_vwap'})
                
        return Signal(action='HOLD', confidence=0.0, meta=meta)