class VolumeSpreadReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.prices = deque(maxlen=15)
        self.volumes = deque(maxlen=15)
        self.bid_ask_history = deque(maxlen=5)
        self.reset()
    
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.bid_ask_history.clear()
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        spread = ask - bid
        mid_price = (bid + ask) / 2.0
        
        self.prices.append(price)
        self.volumes.append(volume)
        self.bid_ask_history.append((bid, ask))
        
        if len(self.prices) < 5:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'initializing'})
        
        avg_volume = sum(self.volumes) / len(self.volumes)
        volume_ratio = volume / (avg_volume + 1e-9)
        
        avg_spread = sum(a - b for b, a in self.bid_ask_history) / len(self.bid_ask_history)
        spread_tightness = avg_spread / (mid_price + 1e-9)
        
        mean_price = sum(self.prices) / len(self.prices)
        deviation = (price - mean_price) / (mean_price + 1e-9)
        
        action = 'HOLD'
        confidence = 0.0
        meta = {
            'deviation': deviation,
            'volume_ratio': volume_ratio,
            'spread_tightness': spread_tightness,
            'mean_price': mean_price
        }
        
        if abs(deviation) > 0.001 and volume_ratio > 1.5 and spread_tightness < 0.001:
            if deviation < 0:
                action = 'BUY'
                confidence = min(0.9, abs(deviation) * 500 + 0.3)
            else:
                action = 'SELL'
                confidence = min(0.9, abs(deviation) * 500 + 0.3)
        
        return Signal(action=action, confidence=confidence, meta=meta)