class SpreadMeanReversionStrategy(BaseStrategy):
    def __init__(self, window: int = 10, velocity_threshold: float = 0.001):
        self.window = window
        self.velocity_threshold = velocity_threshold
        self.prices = deque(maxlen=window)
        self.timestamps = deque(maxlen=window)
        self.spreads = deque(maxlen=window)
        
    def reset(self) -> None:
        self.prices.clear()
        self.timestamps.clear()
        self.spreads.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        timestamp = tick_data.timestamp
        bid = tick_data.bid
        ask = tick_data.ask
        
        current_spread = ask - bid
        
        if len(self.prices) < self.window:
            self.prices.append(price)
            self.timestamps.append(timestamp)
            self.spreads.append(current_spread)
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        time_delta = timestamp - self.timestamps[0]
        if time_delta == 0:
            time_delta = 1
            
        price_velocity = (price - self.prices[0]) / time_delta
        avg_spread = sum(self.spreads) / len(self.spreads)
        spread_tightening = current_spread < avg_spread * 0.9
        
        signal = Signal(action='HOLD', confidence=0.0, meta={})
        
        if spread_tightening:
            if price_velocity < -self.velocity_threshold:
                confidence = min(0.9, abs(price_velocity) * 500)
                signal = Signal(action='BUY', confidence=confidence, meta={
                    'velocity': price_velocity,
                    'spread_compression': avg_spread / (current_spread + 1e-9),
                    'logic': 'oversold_bounce'
                })
            elif price_velocity > self.velocity_threshold:
                confidence = min(0.9, abs(price_velocity) * 500)
                signal = Signal(action='SELL', confidence=confidence, meta={
                    'velocity': price_velocity,
                    'spread_compression': avg_spread / (current_spread + 1e-9),
                    'logic': 'overbought_pullback'
                })
        
        self.prices.append(price)
        self.timestamps.append(timestamp)
        self.spreads.append(current_spread)
        
        return signal