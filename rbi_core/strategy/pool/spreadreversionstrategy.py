class SpreadReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.price_queue = []
        self.spread_history = []
        self.max_queue = 10
        
    def reset(self) -> None:
        self.price_queue.clear()
        self.spread_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        bid = tick_data.bid
        ask = tick_data.ask
        price = tick_data.price
        volume = tick_data.volume
        
        spread = ask - bid
        if spread <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        mid_price = (bid + ask) / 2
        relative_position = (price - bid) / spread
        
        self.price_queue.append(price)
        self.spread_history.append(spread)
        
        if len(self.price_queue) > self.max_queue:
            self.price_queue.pop(0)
            self.spread_history.pop(0)
        
        if len(self.price_queue) < self.max_queue:
            return Signal(action='HOLD', confidence=0.0, meta={'queue_fill': len(self.price_queue)})
        
        avg_spread = sum(self.spread_history) / len(self.spread_history)
        spread_expansion = spread / avg_spread if avg_spread > 0 else 1.0
        
        price_trend = (self.price_queue[-1] - self.price_queue[0]) / self.max_queue
        
        action = 'HOLD'
        confidence = 0.0
        meta = {
            'relative_position': relative_position,
            'spread_expansion': spread_expansion,
            'price_trend': price_trend,
            'volume': volume
        }
        
        if relative_position > 0.75 and spread_expansion > 1.1 and price_trend > 0:
            action = 'BUY'
            confidence = min(relative_position * 0.5 + (spread_expansion - 1) * 0.3 + 0.2, 1.0)
        elif relative_position < 0.25 and spread_expansion > 1.1 and price_trend < 0:
            action = 'SELL'
            confidence = min((1 - relative_position) * 0.5 + (spread_expansion - 1) * 0.3 + 0.2, 1.0)
        
        return Signal(action=action, confidence=confidence, meta=meta)