class BollingerSpreadReversion(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.prices = deque(maxlen=30)
        self.spreads = deque(maxlen=20)
        self.z_threshold = 2.0
        
    def reset(self) -> None:
        self.prices.clear()
        self.spreads.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        
        spread = ask - bid
        spread_pct = (spread / price) * 100 if price > 0 else 0
        
        if len(self.prices) < self.prices.maxlen:
            self.prices.append(price)
            self.spreads.append(spread_pct)
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        n = len(self.prices)
        mean = sum(self.prices) / n
        variance = sum((p - mean) ** 2 for p in self.prices) / n
        std = math.sqrt(variance) if variance > 0 else 0
        
        avg_spread = sum(self.spreads) / len(self.spreads)
        
        self.prices.append(price)
        self.spreads.append(spread_pct)
        
        if std == 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        z_score = (price - mean) / std
        
        if abs(z_score) >= self.z_threshold and spread_pct <= avg_spread * 1.1:
            confidence = min(0.9, abs(z_score) / 3.0)
            if z_score > 0:
                return Signal(
                    action='SELL', 
                    confidence=confidence, 
                    meta={'z_score': z_score, 'mean': mean, 'std': std}
                )
            else:
                return Signal(
                    action='BUY', 
                    confidence=confidence, 
                    meta={'z_score': z_score, 'mean': mean, 'std': std}
                )
        
        return Signal(action='HOLD', confidence=0.0, meta={'z_score': z_score})