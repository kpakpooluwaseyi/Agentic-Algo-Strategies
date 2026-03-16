class BidAskZScoreStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.prices: Deque[float] = deque(maxlen=30)
        self.spreads: Deque[float] = deque(maxlen=30)
        self.last_signal_tick = 0
        self.min_cooldown = 10
        
    def reset(self) -> None:
        self.prices.clear()
        self.spreads.clear()
        self.last_signal_tick = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        timestamp = tick_data.timestamp
        
        spread = ask - bid
        self.prices.append(price)
        self.spreads.append(spread)
        
        if len(self.prices) < 20:
            return Signal(action='HOLD', confidence=0.0, meta={'samples': len(self.prices)})
            
        if timestamp - self.last_signal_tick < self.min_cooldown:
            return Signal(action='HOLD', confidence=0.0, meta={'cooldown_remaining': self.min_cooldown - (timestamp - self.last_signal_tick)})
            
        mean_price = sum(self.prices) / len(self.prices)
        variance = sum((p - mean_price) ** 2 for p in self.prices) / len(self.prices)
        std_price = variance ** 0.5
        
        mean_spread = sum(self.spreads) / len(self.spreads)
        z_score = (price - mean_price) / std_price if std_price > 0 else 0
        
        spread_expansion = spread > mean_spread * 1.25
        
        if z_score > 2.0 and spread_expansion:
            self.last_signal_tick = timestamp
            return Signal(
                action='SELL',
                confidence=min(0.95, 0.5 + abs(z_score) * 0.08),
                meta={'z_score': round(z_score, 2), 'mean': round(mean_price, 2), 'spread_expansion': True}
            )
        elif z_score < -2.0 and spread_expansion:
            self.last_signal_tick = timestamp
            return Signal(
                action='BUY',
                confidence=min(0.95, 0.5 + abs(z_score) * 0.08),
                meta={'z_score': round(z_score, 2), 'mean': round(mean_price, 2), 'spread_expansion': True}
            )
            
        return Signal(action='HOLD', confidence=0.0, meta={'z_score': round(z_score, 2)})