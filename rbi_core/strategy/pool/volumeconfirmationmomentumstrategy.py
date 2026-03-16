class VolumeConfirmationMomentumStrategy(BaseStrategy):
    def __init__(self, volume_window: int = 10, momentum_threshold: float = 0.3):
        self.volume_window = volume_window
        self.momentum_threshold = momentum_threshold
        self.volume_history: Deque[float] = deque(maxlen=volume_window)
        self.prev_price: Optional[float] = None
        self.cooldown_ticks: int = 0
        
    def reset(self) -> None:
        self.volume_history.clear()
        self.prev_price = None
        self.cooldown_ticks = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        if self.prev_price is None:
            self.prev_price = tick_data.price
            self.volume_history.append(tick_data.volume)
            return None
            
        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1
            self.prev_price = tick_data.price
            self.volume_history.append(tick_data.volume)
            return None
            
        if len(self.volume_history) < self.volume_window:
            self.volume_history.append(tick_data.volume)
            self.prev_price = tick_data.price
            return None
            
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        price_change = tick_data.price - self.prev_price
        normalized_change = price_change / tick_data.atr if tick_data.atr > 0 else 0
        volume_ratio = tick_data.volume / avg_volume if avg_volume > 0 else 1.0
        
        signal: Optional[Signal] = None
        
        if volume_ratio > 1.5 and abs(normalized_change) > self.momentum_threshold:
            if normalized_change > 0:
                signal = Signal(
                    action='BUY',
                    confidence=min(0.9, 0.5 + volume_ratio * 0.1 + abs(normalized_change) * 0.1),
                    meta={'volume_ratio': volume_ratio, 'normalized_change': normalized_change, 'price': tick_data.price}
                )
            else:
                signal = Signal(
                    action='SELL',
                    confidence=min(0.9, 0.5 + volume_ratio * 0.1 + abs(normalized_change) * 0.1),
                    meta={'volume_ratio': volume_ratio, 'normalized_change': normalized_change, 'price': tick_data.price}
                )
            self.cooldown_ticks = 3
            
        self.volume_history.append(tick_data.volume)
        self.prev_price = tick_data.price
        return signal