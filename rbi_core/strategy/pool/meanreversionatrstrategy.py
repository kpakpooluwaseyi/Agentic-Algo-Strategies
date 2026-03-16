class MeanReversionATRStrategy(BaseStrategy):
    def __init__(self, lookback: int = 15, deviation_multiple: float = 1.5, volume_confirm: bool = True):
        self.lookback = lookback
        self.deviation_multiple = deviation_multiple
        self.volume_confirm = volume_confirm
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        self.last_signal_price = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        volume = tick_data.volume
        
        if len(self.prices) < self.lookback:
            self.prices.append(price)
            self.volumes.append(volume)
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        mean_price = sum(self.prices) / len(self.prices)
        avg_volume = sum(self.volumes) / len(self.volumes) if self.volumes else 1
        
        deviation = price - mean_price
        atr_normalized_dev = abs(deviation) / atr if atr > 0 else 0
        
        if atr_normalized_dev > self.deviation_multiple:
            volume_confirmed = (not self.volume_confirm) or (volume > avg_volume * 1.2)
            
            if deviation < 0:
                confidence = min(atr_normalized_dev / (self.deviation_multiple * 2), 1.0)
                if volume_confirmed:
                    confidence = min(confidence * 1.1, 1.0)
                signal = Signal(action='BUY', confidence=confidence, meta={'type': 'oversold', 'deviation_atr': -atr_normalized_dev, 'mean': mean_price})
            else:
                confidence = min(atr_normalized_dev / (self.deviation_multiple * 2), 1.0)
                if volume_confirmed:
                    confidence = min(confidence * 1.1, 1.0)
                signal = Signal(action='SELL', confidence=confidence, meta={'type': 'overbought', 'deviation_atr': atr_normalized_dev, 'mean': mean_price})
        else:
            signal = Signal(action='HOLD', confidence=0.0, meta={'deviation_atr': atr_normalized_dev, 'mean': mean_price})
        
        self.prices.append(price)
        self.volumes.append(volume)
        return signal
    
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.last_signal_price = None