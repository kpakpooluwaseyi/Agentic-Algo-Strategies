class AdaptiveMomentumStrategy(BaseStrategy):
    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window
        self.prices = deque(maxlen=long_window)
        self.volumes = deque(maxlen=long_window)
        self.timestamps = deque(maxlen=long_window)
        self.last_signal_time = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        current_time = tick_data.timestamp
        
        if self.last_signal_time is not None:
            if isinstance(current_time, datetime) and isinstance(self.last_signal_time, datetime):
                if (current_time - self.last_signal_time).seconds < 30:
                    return Signal(action='HOLD', confidence=0.0, meta={'cooldown': True})
        
        self.prices.append(tick_data.price)
        self.volumes.append(tick_data.volume)
        self.timestamps.append(current_time)
        
        if len(self.prices) < self.long_window:
            return Signal(action='HOLD', confidence=0.0, meta={'progress': f'{len(self.prices)}/{self.long_window}'})
        
        short_prices = list(self.prices)[-self.short_window:]
        long_prices = list(self.prices)
        
        short_vol = sum(list(self.volumes)[-self.short_window:])
        long_vol = sum(self.volumes)
        
        short_sma = sum(short_prices) / self.short_window
        long_sma = sum(long_prices) / self.long_window
        
        if long_sma == 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        price_ratio = short_sma / long_sma
        vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0
        
        deviation = abs(price_ratio - 1.0) * 100
        
        if price_ratio > 1.02 and vol_ratio > 1.2:
            confidence = min(1.0, (price_ratio - 1.02) * 25 + (vol_ratio - 1.2) * 0.5)
            self.last_signal_time = current_time
            return Signal(
                action='BUY',
                confidence=confidence,
                meta={'momentum': price_ratio, 'volume_surge': vol_ratio, 'deviation_bp': deviation}
            )
        elif price_ratio < 0.98 and vol_ratio > 1.2:
            confidence = min(1.0, (0.98 - price_ratio) * 25 + (vol_ratio - 1.2) * 0.5)
            self.last_signal_time = current_time
            return Signal(
                action='SELL',
                confidence=confidence,
                meta={'momentum': price_ratio, 'volume_surge': vol_ratio, 'deviation_bp': deviation}
            )
            
        return Signal(action='HOLD', confidence=deviation / 100, meta={'momentum': price_ratio})
    
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.timestamps.clear()
        self.last_signal_time = None