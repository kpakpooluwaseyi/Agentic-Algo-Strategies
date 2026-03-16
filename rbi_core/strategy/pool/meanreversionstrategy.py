class MeanReversionStrategy(BaseStrategy):
    def __init__(self, window: int = 15, std_multiplier: float = 2.0, volume_spike: float = 1.5):
        self.window = window
        self.std_multiplier = std_multiplier
        self.volume_spike = volume_spike
        self.prices: List[float] = []
        self.volumes: List[float] = []
        self.avg_volume: float = 0.0
        
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.avg_volume = 0.0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        if len(self.prices) < self.window:
            self.prices.append(price)
            self.volumes.append(volume)
            if self.volumes:
                self.avg_volume = sum(self.volumes) / len(self.volumes)
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        mean_price = statistics.mean(self.prices)
        std_price = statistics.stdev(self.prices) if len(self.prices) > 1 else 0.0
            
        upper_band = mean_price + (self.std_multiplier * std_price)
        lower_band = mean_price - (self.std_multiplier * std_price)
        
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) > self.window:
            removed_vol = self.volumes.pop(0)
            self.prices.pop(0)
            if self.volumes:
                self.avg_volume = (self.avg_volume * (self.window - 1) - removed_vol + volume) / (self.window - 1)
        
        volume_ratio = volume / self.avg_volume if self.avg_volume > 0 else 1.0
        
        meta = {
            'mean': round(mean_price, 4),
            'std': round(std_price, 4),
            'upper_band': round(upper_band, 4),
            'lower_band': round(lower_band, 4),
            'volume_ratio': round(volume_ratio, 2)
        }
        
        if price <= lower_band and volume_ratio > self.volume_spike:
            confidence = min(1.0, (lower_band - price) / (std_price + 0.0001) + (volume_ratio - 1) * 0.2)
            return Signal(action='BUY', confidence=round(confidence, 2), meta=meta)
        
        elif price >= upper_band and volume_ratio > self.volume_spike:
            confidence = min(1.0, (price - upper_band) / (std_price + 0.0001) + (volume_ratio - 1) * 0.2)
            return Signal(action='SELL', confidence=round(confidence, 2), meta=meta)
        
        return Signal(action='HOLD', confidence=0.0, meta=meta)