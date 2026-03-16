class VolumeWeightedRSIStrategy(BaseStrategy):
    def __init__(self, rsi_period: int = 14, volume_period: int = 20):
        super().__init__()
        self.rsi_period = rsi_period
        self.volume_period = volume_period
        self.price_history = deque(maxlen=rsi_period)
        self.volume_history = deque(maxlen=volume_period)
        self.prev_price = None
        self.avg_gain = None
        self.avg_loss = None
        self.initialized = False
        
    def reset(self) -> None:
        self.price_history.clear()
        self.volume_history.clear()
        self.prev_price = None
        self.avg_gain = None
        self.avg_loss = None
        self.initialized = False
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        self.volume_history.append(volume)
        
        if self.prev_price is None:
            self.prev_price = price
            return None
            
        change = price - self.prev_price
        gain = change if change > 0 else 0
        loss = abs(change) if change < 0 else 0
        
        if not self.initialized:
            self.price_history.append((gain, loss))
            if len(self.price_history) < self.rsi_period:
                self.prev_price = price
                return None
            self.avg_gain = sum(g[0] for g in self.price_history) / self.rsi_period
            self.avg_loss = sum(l[1] for l in self.price_history) / self.rsi_period
            self.initialized = True
        else:
            self.avg_gain = (self.avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self.avg_loss = (self.avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period
            
        if self.avg_loss == 0:
            rsi = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
        avg_volume = sum(self.volume_history) / len(self.volume_history) if self.volume_history else volume
        volume_factor = min(volume / avg_volume, 2.0) if avg_volume > 0 else 1.0
        
        if rsi < 30:
            confidence = min((30.0 - rsi) / 30.0 * 0.5 * volume_factor + 0.5, 0.95)
            signal = Signal(action='BUY', confidence=confidence, meta={'rsi': rsi, 'volume_factor': volume_factor})
        elif rsi > 70:
            confidence = min((rsi - 70.0) / 30.0 * 0.5 * volume_factor + 0.5, 0.95)
            signal = Signal(action='SELL', confidence=confidence, meta={'rsi': rsi, 'volume_factor': volume_factor})
        else:
            signal = Signal(action='HOLD', confidence=0.5, meta={'rsi': rsi, 'volume_factor': volume_factor})
            
        self.prev_price = price
        return signal