class VolumeConfirmedMeanReversion(BaseStrategy):
    def __init__(self, lookback: int = 10, volume_threshold: float = 1.2, deviation_factor: float = 1.5):
        super().__init__()
        self.lookback = lookback
        self.volume_threshold = volume_threshold
        self.deviation_factor = deviation_factor
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        self.atrs = deque(maxlen=lookback)
    
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.atrs.clear()
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        if len(self.prices) < self.lookback:
            self.prices.append(price)
            self.volumes.append(volume)
            self.atrs.append(atr)
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'accumulating'})
        
        sma = sum(self.prices) / len(self.prices)
        avg_volume = sum(self.volumes) / len(self.volumes)
        avg_atr = sum(self.atrs) / len(self.atrs)
        
        volume_surge = volume > (avg_volume * self.volume_threshold)
        upper_band = sma + (self.deviation_factor * avg_atr)
        lower_band = sma - (self.deviation_factor * avg_atr)
        
        if price < lower_band and volume_surge:
            deviation = (lower_band - price) / avg_atr if avg_atr > 0 else 0
            confidence = min(0.5 + (deviation * 0.25), 1.0)
            signal = Signal(
                action='BUY',
                confidence=confidence,
                meta={'type': 'oversold_reversal', 'sma': sma, 'volume_ratio': volume/avg_volume, 'deviation': deviation}
            )
        elif price > upper_band and volume_surge:
            deviation = (price - upper_band) / avg_atr if avg_atr > 0 else 0
            confidence = min(0.5 + (deviation * 0.25), 1.0)
            signal = Signal(
                action='SELL',
                confidence=confidence,
                meta={'type': 'overbought_reversal', 'sma': sma, 'volume_ratio': volume/avg_volume, 'deviation': deviation}
            )
        else:
            signal = Signal(
                action='HOLD',
                confidence=0.0,
                meta={'sma': sma, 'distance_from_mean': (price - sma) / avg_atr if avg_atr > 0 else 0, 'volume_surge': volume_surge}
            )
        
        self.prices.append(price)
        self.volumes.append(volume)
        self.atrs.append(atr)
        return signal