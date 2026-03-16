class TimeBasedMomentumStrategy(BaseStrategy):
    def __init__(self, fast_period: int = 5, slow_period: int = 12, hold_timeout: float = 30.0):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.hold_timeout = hold_timeout
        self.price_history: List[float] = []
        self.time_history: List[float] = []
        self.entry_time: Optional[float] = None
        self.entry_price: Optional[float] = None
        self.side = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        current_price = tick_data.price
        current_time = tick_data.timestamp
        
        self.price_history.append(current_price)
        self.time_history.append(current_time)
        
        while len(self.price_history) > self.slow_period:
            self.price_history.pop(0)
            self.time_history.pop(0)
            
        if len(self.price_history) < self.slow_period:
            return Signal(action='HOLD', confidence=0.0, meta={'accumulating': len(self.price_history)})
        
        fast_ma = sum(self.price_history[-self.fast_period:]) / self.fast_period
        slow_ma = sum(self.price_history) / self.slow_period
        
        action = 'HOLD'
        confidence = 0.0
        
        time_in_trade = 0.0
        if self.entry_time is not None:
            time_in_trade = current_time - self.entry_time
            
        meta = {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'time_in_trade': time_in_trade,
            'side': self.side
        }
        
        if self.side != 0 and time_in_trade > self.hold_timeout:
            action = 'SELL' if self.side == 1 else 'BUY'
            confidence = 0.6
            self.side = 0
            self.entry_time = None
            self.entry_price = None
            return Signal(action=action, confidence=confidence, meta={**meta, 'reason': 'timeout'})
        
        if fast_ma > slow_ma * 1.002 and self.side != 1:
            action = 'BUY'
            confidence = min(0.95, (fast_ma / slow_ma - 1) * 200)
            self.side = 1
            self.entry_time = current_time
            self.entry_price = current_price
        elif fast_ma < slow_ma * 0.998 and self.side != -1:
            action = 'SELL'
            confidence = min(0.95, (1 - fast_ma / slow_ma) * 200)
            self.side = -1
            self.entry_time = current_time
            self.entry_price = current_price
        
        if self.entry_price and tick_data.atr > 0:
            loss_pct = abs(current_price - self.entry_price) / tick_data.atr
            if loss_pct > 1.5:
                if (self.side == 1 and current_price < self.entry_price) or (self.side == -1 and current_price > self.entry_price):
                    action = 'SELL' if self.side == 1 else 'BUY'
                    confidence = 0.8
                    self.side = 0
                    self.entry_time = None
                    self.entry_price = None
                    meta['reason'] = 'stop_loss'
        
        return Signal(action=action, confidence=confidence, meta=meta)
    
    def reset(self) -> None:
        self.price_history.clear()
        self.time_history.clear()
        self.entry_time = None
        self.entry_price = None
        self.side = 0