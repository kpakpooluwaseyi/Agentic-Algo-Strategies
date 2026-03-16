class SpreadMomentumStrategy(BaseStrategy):
    def __init__(self):
        self.spread_history = deque(maxlen=10)
        self.price_history = deque(maxlen=10)
        self.timestamp_history = deque(maxlen=10)
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        mid_price = (tick_data.bid + tick_data.ask) / 2
        spread = tick_data.ask - tick_data.bid
        timestamp = tick_data.timestamp
        
        self.spread_history.append(spread)
        self.price_history.append(mid_price)
        self.timestamp_history.append(timestamp)
        
        if len(self.spread_history) < 10:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        avg_spread = sum(self.spread_history) / len(self.spread_history)
        
        if len(self.price_history) >= 2:
            price_diff = mid_price - self.price_history[-2]
            time_diff = timestamp - self.timestamp_history[-2]
            velocity = price_diff / time_diff if time_diff > 0 else 0
        else:
            velocity = 0
        
        meta = {
            'spread': spread,
            'avg_spread': avg_spread,
            'velocity': velocity,
            'mid_price': mid_price
        }
        
        if spread < avg_spread * 0.8 and velocity > 0:
            confidence = min(1.0, 0.6 + abs(velocity) * 10)
            return Signal(action='BUY', confidence=confidence, meta=meta)
        elif spread > avg_spread * 1.2 and velocity < 0:
            confidence = min(1.0, 0.6 + abs(velocity) * 10)
            return Signal(action='SELL', confidence=confidence, meta=meta)
        
        return Signal(action='HOLD', confidence=0.0, meta=meta)
    
    def reset(self) -> None:
        self.spread_history.clear()
        self.price_history.clear()
        self.timestamp_history.clear()