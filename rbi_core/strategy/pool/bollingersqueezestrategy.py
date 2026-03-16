class BollingerSqueezeStrategy(BaseStrategy):
    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__()
        self.period = period
        self.num_std = num_std
        self.prices: Deque[float] = deque(maxlen=period)
        self.prev_band: int = 0
        
    def reset(self) -> None:
        self.prices.clear()
        self.prev_band = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        self.prices.append(price)
        
        if len(self.prices) < self.period:
            return None
            
        sma = mean(self.prices)
        std_dev = stdev(self.prices)
        
        upper_band = sma + (self.num_std * std_dev)
        lower_band = sma - (self.num_std * std_dev)
        
        current_band = 0
        if price > upper_band:
            current_band = 1
        elif price < lower_band:
            current_band = -1
            
        signal = None
        if current_band == 1 and self.prev_band != 1:
            deviation = (price - upper_band) / (std_dev + 1e-9)
            confidence = min(0.5 + deviation * 0.25, 1.0)
            signal = Signal(action='SELL', confidence=confidence,
                          meta={'sma': sma, 'upper': upper_band, 'lower': lower_band, 'deviation': deviation})
        elif current_band == -1 and self.prev_band != -1:
            deviation = (lower_band - price) / (std_dev + 1e-9)
            confidence = min(0.5 + deviation * 0.25, 1.0)
            signal = Signal(action='BUY', confidence=confidence,
                          meta={'sma': sma, 'upper': upper_band, 'lower': lower_band, 'deviation': deviation})
        else:
            signal = Signal(action='HOLD', confidence=0.0,
                          meta={'sma': sma, 'position': 'within_bands'})
            
        self.prev_band = current_band
        return signal