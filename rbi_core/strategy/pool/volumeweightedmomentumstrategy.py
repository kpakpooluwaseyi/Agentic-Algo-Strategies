class VolumeWeightedMomentumStrategy(BaseStrategy):
    def __init__(self):
        self.prices = []
        self.volumes = []
        self.max_lookback = 15
        self.momentum_threshold = 0.01
        
    def reset(self) -> None:
        self.prices = []
        self.volumes = []
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        current_price = tick_data.price
        current_volume = tick_data.volume
        
        self.prices.append(current_price)
        self.volumes.append(current_volume)
        
        if len(self.prices) > self.max_lookback:
            self.prices.pop(0)
            self.volumes.pop(0)
            
        if len(self.prices) < self.max_lookback:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        weighted_sum = sum(p * v for p, v in zip(self.prices, self.volumes))
        total_volume = sum(self.volumes)
        
        if total_volume == 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        vwap = weighted_sum / total_volume
        price_change = (current_price - self.prices[0]) / self.prices[0] if self.prices[0] != 0 else 0
        vwap_deviation = (current_price - vwap) / vwap if vwap != 0 else 0
        
        if price_change > self.momentum_threshold and vwap_deviation > 0:
            confidence = min(abs(price_change) * 50, 1.0)
            return Signal(action='BUY', confidence=confidence, meta={'vwap': vwap, 'momentum': price_change})
        elif price_change < -self.momentum_threshold and vwap_deviation < 0:
            confidence = min(abs(price_change) * 50, 1.0)
            return Signal(action='SELL', confidence=confidence, meta={'vwap': vwap, 'momentum': price_change})
            
        return Signal(action='HOLD', confidence=0.0, meta={})