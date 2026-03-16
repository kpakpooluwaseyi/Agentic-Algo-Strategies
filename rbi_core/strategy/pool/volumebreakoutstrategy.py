class VolumeBreakoutStrategy(BaseStrategy):
    def __init__(self):
        self.volume_history = deque(maxlen=15)
        self.price_history = deque(maxlen=5)
        self.atr_threshold = 0.0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        self.volume_history.append(volume)
        self.price_history.append(price)
        
        if len(self.volume_history) < 15:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        volume_surge = volume / avg_volume if avg_volume > 0 else 1.0
        
        if len(self.price_history) >= 2:
            prev_price = self.price_history[-2]
            price_change = (price - prev_price) / prev_price if prev_price != 0 else 0
        else:
            price_change = 0
        
        meta = {
            'volume_surge': volume_surge,
            'price_change_pct': price_change * 100,
            'avg_volume': avg_volume,
            'current_atr': atr
        }
        
        if volume_surge > 2.5 and price_change > 0.0005:
            confidence = min(1.0, (volume_surge - 2.5) * 0.4 + abs(price_change) * 500)
            return Signal(action='BUY', confidence=confidence, meta=meta)
        elif volume_surge > 2.5 and price_change < -0.0005:
            confidence = min(1.0, (volume_surge - 2.5) * 0.4 + abs(price_change) * 500)
            return Signal(action='SELL', confidence=confidence, meta=meta)
        
        return Signal(action='HOLD', confidence=0.0, meta=meta)
    
    def reset(self) -> None:
        self.volume_history.clear()
        self.price_history.clear()
        self.atr_threshold = 0.0