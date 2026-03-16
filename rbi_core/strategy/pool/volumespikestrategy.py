class VolumeSpikeStrategy(BaseStrategy):
    def __init__(self):
        self.volume_history = deque(maxlen=20)
        self.price_history = deque(maxlen=20)
        
    def reset(self) -> None:
        self.volume_history.clear()
        self.price_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        if len(self.volume_history) < 10:
            self.volume_history.append(volume)
            self.price_history.append(price)
            return None
            
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        vol_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        prev_price = self.price_history[-1]
        price_change_pct = (price - prev_price) / prev_price if prev_price != 0 else 0
        
        self.volume_history.append(volume)
        self.price_history.append(price)
        
        if vol_ratio > 2.0:
            if price_change_pct > 0.001:
                return Signal(action='BUY', confidence=min(vol_ratio / 4.0, 1.0), meta={'volume_ratio': vol_ratio, 'momentum': 'positive'})
            elif price_change_pct < -0.001:
                return Signal(action='SELL', confidence=min(vol_ratio / 4.0, 1.0), meta={'volume_ratio': vol_ratio, 'momentum': 'negative'})
                
        return Signal(action='HOLD', confidence=0.3, meta={'volume_ratio': vol_ratio})