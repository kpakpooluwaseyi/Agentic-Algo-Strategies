class MomentumBreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.prev_price = None
        self.prev_atr = None
        self.consecutive_bars = 0
        self.history = []
        self.lookback = 5
        
    def reset(self) -> None:
        self.prev_price = None
        self.prev_atr = None
        self.consecutive_bars = 0
        self.history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        timestamp = tick_data.timestamp
        
        self.history.append((timestamp, price, atr))
        if len(self.history) > self.lookback:
            self.history.pop(0)
            
        if len(self.history) < self.lookback:
            self.prev_price = price
            self.prev_atr = atr
            return None
            
        if self.prev_price is None or self.prev_price == 0:
            self.prev_price = price
            self.prev_atr = atr
            return None
            
        price_change = (price - self.prev_price) / self.prev_price
        atr_rising = atr > self.prev_atr * 1.02
        
        oldest_price = self.history[0][1]
        total_range = (price - oldest_price) / oldest_price if oldest_price != 0 else 0
        
        meta = {
            'price_change_pct': price_change * 100,
            'atr_rising': atr_rising,
            'total_range_pct': total_range * 100
        }
        
        if price_change > 0 and atr_rising:
            self.consecutive_bars = min(self.consecutive_bars + 1, 5)
            confidence = min(0.95, 0.5 + abs(total_range) * 10 + self.consecutive_bars * 0.08)
            signal = Signal(action='BUY', confidence=confidence, meta=meta)
        elif price_change < 0 and atr_rising:
            self.consecutive_bars = min(self.consecutive_bars + 1, 5)
            confidence = min(0.95, 0.5 + abs(total_range) * 10 + self.consecutive_bars * 0.08)
            signal = Signal(action='SELL', confidence=confidence, meta=meta)
        else:
            self.consecutive_bars = max(0, self.consecutive_bars - 1)
            signal = Signal(action='HOLD', confidence=0.15, meta=meta)
            
        self.prev_price = price
        self.prev_atr = atr
        return signal