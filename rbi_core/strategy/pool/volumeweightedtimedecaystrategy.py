class VolumeWeightedTimeDecayStrategy(BaseStrategy):
    def __init__(self, short_period: int = 5, long_period: int = 15, decay_lambda: float = 0.01):
        self.short_period = short_period
        self.long_period = long_period
        self.decay_lambda = decay_lambda
        self.price_volume_short = deque(maxlen=short_period)
        self.price_volume_long = deque(maxlen=long_period)
        self.volume_short = deque(maxlen=short_period)
        self.volume_long = deque(maxlen=long_period)
        self.last_timestamp = None
        self.cumulative_time = 0
        
    def reset(self) -> None:
        self.price_volume_short.clear()
        self.price_volume_long.clear()
        self.volume_short.clear()
        self.volume_long.clear()
        self.last_timestamp = None
        self.cumulative_time = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        current_time = tick_data.timestamp
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        if self.last_timestamp is not None:
            time_delta = current_time - self.last_timestamp
            self.cumulative_time += time_delta
        self.last_timestamp = current_time
        
        pv = price * volume
        self.price_volume_short.append(pv)
        self.price_volume_long.append(pv)
        self.volume_short.append(volume)
        self.volume_long.append(volume)
        
        if len(self.volume_long) < self.long_period:
            return Signal(action='HOLD', confidence=0.0, meta={'warming': f'{len(self.volume_long)}/{self.long_period}'})
        
        vwap_short = sum(self.price_volume_short) / sum(self.volume_short) if sum(self.volume_short) > 0 else price
        vwap_long = sum(self.price_volume_long) / sum(self.volume_long) if sum(self.volume_long) > 0 else price
        
        time_decay = math.exp(-self.decay_lambda * self.cumulative_time)
        vwap_diff = (vwap_short - vwap_long) / vwap_long if vwap_long != 0 else 0
        adjusted_diff = vwap_diff * time_decay
        
        atr_percent = (atr / price) if price > 0 else 0
        
        if adjusted_diff > 0.0008 + atr_percent * 0.5:
            confidence = min(1.0, abs(adjusted_diff) * 100 + (1 - time_decay) * 0.3)
            return Signal(action='BUY', confidence=confidence, meta={'vwap_diff': vwap_diff, 'decay': time_decay, 'vwap_s': vwap_short, 'vwap_l': vwap_long})
        elif adjusted_diff < -0.0008 - atr_percent * 0.5:
            confidence = min(1.0, abs(adjusted_diff) * 100 + (1 - time_decay) * 0.3)
            return Signal(action='SELL', confidence=confidence, meta={'vwap_diff': vwap_diff, 'decay': time_decay, 'vwap_s': vwap_short, 'vwap_l': vwap_long})
        
        return Signal(action='HOLD', confidence=abs(adjusted_diff) * 50 if abs(adjusted_diff) > 0 else 0.0, meta={'vwap_diff': adjusted_diff, 'time_decay': time_decay})