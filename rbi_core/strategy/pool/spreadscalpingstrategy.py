class SpreadScalpingStrategy(BaseStrategy):
    def __init__(self, spread_threshold=0.0005, atr_filter_factor=0.5):
        super().__init__()
        self.spread_threshold = spread_threshold
        self.atr_filter_factor = atr_filter_factor
        self.recent_mid_prices = deque(maxlen=5)
        self.prev_bid = None
        self.prev_ask = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        mid_price = (tick_data.bid + tick_data.ask) / 2
        spread = tick_data.ask - tick_data.bid
        self.recent_mid_prices.append(mid_price)
        
        if self.prev_bid is None:
            self.prev_bid = tick_data.bid
            self.prev_ask = tick_data.ask
            return Signal(action='HOLD', confidence=0.0, meta={'init': True})
            
        spread_pct = spread / mid_price if mid_price > 0 else 0
        
        if spread_pct < self.spread_threshold:
            self.prev_bid = tick_data.bid
            self.prev_ask = tick_data.ask
            return Signal(action='HOLD', confidence=0.0, meta={'spread_too_narrow': spread_pct})
            
        bid_pressure = tick_data.bid - self.prev_bid
        ask_pressure = tick_data.ask - self.prev_ask
        
        if len(self.recent_mid_prices) < 5:
            self.prev_bid = tick_data.bid
            self.prev_ask = tick_data.ask
            return Signal(action='HOLD', confidence=0.0, meta={'warming_up': True})
            
        short_trend = (mid_price - self.recent_mid_prices[0]) / self.recent_mid_prices[0] if self.recent_mid_prices[0] > 0 else 0
        volatility_ok = tick_data.atr < mid_price * self.atr_filter_factor
        
        action = 'HOLD'
        confidence = 0.0
        
        if bid_pressure > 0 and ask_pressure > 0 and short_trend > 0.0001 and volatility_ok:
            if tick_data.price < mid_price:
                action = 'BUY'
                confidence = 0.6 + min(0.35, abs(short_trend) * 1000)
        elif bid_pressure < 0 and ask_pressure < 0 and short_trend < -0.0001 and volatility_ok:
            if tick_data.price > mid_price:
                action = 'SELL'
                confidence = 0.6 + min(0.35, abs(short_trend) * 1000)
                
        self.prev_bid = tick_data.bid
        self.prev_ask = tick_data.ask
        
        return Signal(
            action=action,
            confidence=confidence,
            meta={
                'spread_pct': spread_pct,
                'bid_pressure': bid_pressure,
                'ask_pressure': ask_pressure,
                'short_trend': short_trend,
                'volatility_ok': volatility_ok
            }
        )
    
    def reset(self) -> None:
        self.recent_mid_prices.clear()
        self.prev_bid = None
        self.prev_ask = None