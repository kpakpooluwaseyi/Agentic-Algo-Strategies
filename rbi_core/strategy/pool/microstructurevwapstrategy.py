class MicrostructureVwapStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20):
        super().__init__()
        self.lookback = lookback
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
    
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        bid = tick_data.bid
        ask = tick_data.ask
        price = tick_data.price
        volume = tick_data.volume
        
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        spread = ask - bid
        mid = (bid + ask) / 2
        
        vwap_numerator = sum(p * v for p, v in zip(self.prices, self.volumes))
        vwap_denominator = sum(self.volumes)
        vwap = vwap_numerator / vwap_denominator if vwap_denominator != 0 else price
        
        avg_volume = sum(self.volumes) / len(self.volumes)
        
        action = 'HOLD'
        confidence = 0.0
        
        if price <= bid + spread * 0.2 and price < vwap and volume > avg_volume * 1.5:
            action = 'BUY'
            confidence = min(1.0, (vwap - price) / (spread + 1e-9) * 0.3)
        elif price >= ask - spread * 0.2 and price > vwap and volume > avg_volume * 1.5:
            action = 'SELL'
            confidence = min(1.0, (price - vwap) / (spread + 1e-9) * 0.3)
        
        return Signal(action=action, confidence=confidence, meta={'vwap': vwap, 'spread': spread, 'avg_volume': avg_volume})