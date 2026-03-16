class DualEMAStrategy(BaseStrategy):
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_alpha = 2.0 / (fast_period + 1)
        self.slow_alpha = 2.0 / (slow_period + 1)
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        
    def reset(self) -> None:
        self.fast_ema = None
        self.slow_ema = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        
        if self.fast_ema is None:
            self.fast_ema = price
            self.slow_ema = price
            return None
            
        prev_fast = self.fast_ema
        prev_slow = self.slow_ema
        
        self.fast_ema = (self.fast_alpha * price) + ((1 - self.fast_alpha) * prev_fast)
        self.slow_ema = (self.slow_alpha * price) + ((1 - self.slow_alpha) * prev_slow)
        
        signal = None
        if prev_fast <= prev_slow and self.fast_ema > self.slow_ema:
            confidence = min(abs(self.fast_ema - self.slow_ema) / (tick_data.atr + 1e-9), 1.0)
            signal = Signal(action='BUY', confidence=confidence, 
                          meta={'fast_ema': self.fast_ema, 'slow_ema': self.slow_ema, 'cross': 'golden'})
        elif prev_fast >= prev_slow and self.fast_ema < self.slow_ema:
            confidence = min(abs(self.fast_ema - self.slow_ema) / (tick_data.atr + 1e-9), 1.0)
            signal = Signal(action='SELL', confidence=confidence,
                          meta={'fast_ema': self.fast_ema, 'slow_ema': self.slow_ema, 'cross': 'death'})
        else:
            signal = Signal(action='HOLD', confidence=0.0, 
                          meta={'fast_ema': self.fast_ema, 'slow_ema': self.slow_ema})
            
        return signal