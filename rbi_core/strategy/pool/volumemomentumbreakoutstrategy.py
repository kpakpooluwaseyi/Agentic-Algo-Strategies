class VolumeMomentumBreakoutStrategy(BaseStrategy):
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        self.prev_price: Optional[float] = None
        self.volume_history: List[float] = []
        self.price_changes: List[float] = []
        self.vpt = 0.0
        self.window = 15
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        if self.prev_price is None:
            self.prev_price = price
            return Signal(action='HOLD', confidence=0.0, meta={'vpt': 0.0})
        
        if self.prev_price != 0:
            pct_change = (price - self.prev_price) / self.prev_price
        else:
            pct_change = 0.0
        
        self.vpt += volume * pct_change
        self.price_changes.append(pct_change)
        self.volume_history.append(volume)
        
        if len(self.price_changes) > self.window:
            self.price_changes.pop(0)
            self.volume_history.pop(0)
        
        self.prev_price = price
        
        if len(self.volume_history) < self.window:
            return Signal(action='HOLD', confidence=0.0, meta={'vpt': self.vpt})
        
        avg_volume = sum(self.volume_history) / self.window
        cumulative_momentum = sum(self.price_changes)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio > 1.5:
            if cumulative_momentum > 0.015:
                confidence = min(1.0, cumulative_momentum * 30 + (volume_ratio - 1.5) * 0.2)
                return Signal(
                    action='BUY',
                    confidence=confidence,
                    meta={'vpt': self.vpt, 'momentum': cumulative_momentum, 'vol_ratio': volume_ratio}
                )
            elif cumulative_momentum < -0.015:
                confidence = min(1.0, abs(cumulative_momentum) * 30 + (volume_ratio - 1.5) * 0.2)
                return Signal(
                    action='SELL',
                    confidence=confidence,
                    meta={'vpt': self.vpt, 'momentum': cumulative_momentum, 'vol_ratio': volume_ratio}
                )
        
        return Signal(action='HOLD', confidence=0.0, meta={'vpt': self.vpt, 'momentum': cumulative_momentum})