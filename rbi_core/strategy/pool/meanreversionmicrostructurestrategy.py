class MeanReversionMicrostructureStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.price_history: List[float] = []
        self.spread_history: List[float] = []
        self.max_lookback = 15
        self.z_threshold = 2.0
        
    def reset(self) -> None:
        self.price_history.clear()
        self.spread_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        
        spread = ask - bid
        mid_price = (ask + bid) / 2
        
        self.price_history.append(price)
        self.spread_history.append(spread)
        
        if len(self.price_history) > self.max_lookback:
            self.price_history.pop(0)
            self.spread_history.pop(0)
            
        if len(self.price_history) < 8:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'warming_up'})
        
        mean_price = statistics.mean(self.price_history)
        std_price = statistics.stdev(self.price_history) if len(self.price_history) > 1 else 0.0001
        avg_spread = statistics.mean(self.spread_history)
        
        z_score = (price - mean_price) / std_price if std_price > 0 else 0
        spread_tightness = spread / avg_spread if avg_spread > 0 else 1.0
        
        trade_location = (price - bid) / spread if spread > 0 else 0.5
        
        meta = {
            'z_score': z_score,
            'spread_tightness': spread_tightness,
            'trade_location': trade_location
        }
        
        if abs(z_score) > self.z_threshold and spread_tightness < 1.5:
            if z_score > 0 and trade_location > 0.6:
                confidence = min(0.9, 0.5 + (z_score - self.z_threshold) * 0.1)
                return Signal(action='SELL', confidence=confidence, meta=meta)
            elif z_score < 0 and trade_location < 0.4:
                confidence = min(0.9, 0.5 + (abs(z_score) - self.z_threshold) * 0.1)
                return Signal(action='BUY', confidence=confidence, meta=meta)
                
        return Signal(action='HOLD', confidence=0.0, meta=meta)