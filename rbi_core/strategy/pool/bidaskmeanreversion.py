class BidAskMeanReversion(BaseStrategy):
    def __init__(self, atr_window: int = 15, volatility_percentile: float = 0.3, extreme_threshold: float = 0.2):
        self.atr_window = atr_window
        self.volatility_percentile = volatility_percentile
        self.extreme_threshold = extreme_threshold
        self.atr_history = deque(maxlen=atr_window)
    
    def reset(self) -> None:
        self.atr_history.clear()
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        atr = tick_data.atr
        
        if ask <= bid:
            return Signal(action='HOLD', confidence=0.0, meta={'invalid_spread': True})
        
        self.atr_history.append(atr)
        
        if len(self.atr_history) < self.atr_window:
            return Signal(action='HOLD', confidence=0.0, meta={'atr_warmup': len(self.atr_history)})
        
        sorted_atr = sorted(self.atr_history)
        low_volatility_threshold = sorted_atr[int(self.atr_window * self.volatility_percentile)]
        
        if atr > low_volatility_threshold:
            return Signal(action='HOLD', confidence=0.0, meta={'volatility_filter': 'too_high'})
        
        relative_position = (price - bid) / (ask - bid)
        
        meta = {
            'rel_pos': relative_position,
            'atr_limit': low_volatility_threshold
        }
        
        if relative_position < self.extreme_threshold:
            confidence = (self.extreme_threshold - relative_position) / self.extreme_threshold
            return Signal(action='BUY', confidence=confidence, meta=meta)
        elif relative_position > (1.0 - self.extreme_threshold):
            confidence = (relative_position - (1.0 - self.extreme_threshold)) / self.extreme_threshold
            return Signal(action='SELL', confidence=confidence, meta=meta)
        else:
            return Signal(action='HOLD', confidence=0.0, meta=meta)