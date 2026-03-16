class BidAskMicrostructure(BaseStrategy):
    def __init__(self, momentum_window: int = 5, spread_threshold: float = 0.0001):
        self.momentum_window = momentum_window
        self.spread_threshold = spread_threshold
        self.prices = deque(maxlen=momentum_window)
        self.mid_prices = deque(maxlen=momentum_window)
        
    def reset(self) -> None:
        self.prices.clear()
        self.mid_prices.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        
        if ask <= bid:
            return Signal(action='HOLD', confidence=0.0, meta={'invalid_spread': True})
        
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid
        
        self.prices.append(price)
        self.mid_prices.append(mid)
        
        if len(self.prices) < self.momentum_window:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'accumulating'})
        
        price_velocity = price - self.prices[0]
        mid_velocity = mid - self.mid_prices[0]
        
        action = 'HOLD'
        confidence = 0.0
        
        if spread_pct > self.spread_threshold:
            if price > mid and price_velocity > 0 and mid_velocity > 0:
                action = 'BUY'
                confidence = 0.7 + 0.2 * min(1.0, price_velocity / (mid * 0.001))
            elif price < mid and price_velocity < 0 and mid_velocity < 0:
                action = 'SELL'
                confidence = 0.7 + 0.2 * min(1.0, abs(price_velocity) / (mid * 0.001))
        
        return Signal(action=action, confidence=confidence, meta={
            'spread_pct': spread_pct,
            'price_velocity': price_velocity,
            'mid': mid,
            'position_in_spread': (price - bid) / (ask - bid)
        })