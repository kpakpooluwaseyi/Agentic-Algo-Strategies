class VolumeSurgeMomentumStrategy(BaseStrategy):
    def __init__(self, lookback: int = 10, volume_threshold: float = 2.0, min_price_change_pct: float = 0.0005):
        self.lookback = lookback
        self.volume_threshold = volume_threshold
        self.min_price_change_pct = min_price_change_pct
        self.prices: List[float] = []
        self.volumes: List[float] = []
        
    def reset(self) -> None:
        self.prices = []
        self.volumes = []
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) > self.lookback:
            self.prices.pop(0)
            self.volumes.pop(0)
            
        if len(self.volumes) < self.lookback:
            return None
            
        avg_volume = sum(self.volumes[:-1]) / (len(self.volumes) - 1)
        prev_price = self.prices[-2] if len(self.prices) >= 2 else price
        price_change_pct = (price - prev_price) / prev_price if prev_price != 0 else 0
        
        volume_ratio = volume / (avg_volume + 1e-9)
        
        meta = {
            'volume_ratio': volume_ratio,
            'price_change_pct': price_change_pct,
            'avg_volume': avg_volume
        }
        
        if volume_ratio > self.volume_threshold:
            if price_change_pct > self.min_price_change_pct:
                confidence = min(1.0, volume_ratio / 4.0)
                return Signal(action='BUY', confidence=confidence, meta=meta)
            elif price_change_pct < -self.min_price_change_pct:
                confidence = min(1.0, volume_ratio / 4.0)
                return Signal(action='SELL', confidence=confidence, meta=meta)
                
        return Signal(action='HOLD', confidence=0.5, meta=meta)