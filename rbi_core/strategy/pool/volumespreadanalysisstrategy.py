class VolumeSpreadAnalysisStrategy(BaseStrategy):
    def __init__(self, volume_period: int = 10, volume_threshold: float = 1.5):
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        self.volume_history: deque = deque(maxlen=volume_period)
        
    def reset(self) -> None:
        self.volume_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        if len(self.volume_history) < self.volume_period:
            self.volume_history.append(volume)
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'accumulating_volume'})
            
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        is_high_volume = volume_ratio > self.volume_threshold
        
        spread = ask - bid
        if spread > 0:
            position_in_spread = (price - bid) / spread
        else:
            position_in_spread = 0.5
            
        meta = {
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'position_in_spread': position_in_spread,
            'spread': spread
        }
        
        if is_high_volume:
            if position_in_spread > 0.8:
                confidence = min(position_in_spread * (volume_ratio / self.volume_threshold), 1.0)
                self.volume_history.append(volume)
                return Signal(action='BUY', confidence=confidence, meta={**meta, 'trigger': 'high_volume_ask_pressure'})
            elif position_in_spread < 0.2:
                confidence = min((1.0 - position_in_spread) * (volume_ratio / self.volume_threshold), 1.0)
                self.volume_history.append(volume)
                return Signal(action='SELL', confidence=confidence, meta={**meta, 'trigger': 'high_volume_bid_pressure'})
                
        self.volume_history.append(volume)
        return Signal(action='HOLD', confidence=0.0, meta=meta)