class VWAPMeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.max_lookback = 50
        self.volume_trend_window = 10
        self.deviation_threshold = 2.0
        
        self.cumulative_pv = 0.0
        self.cumulative_volume = 0.0
        self.price_history = deque(maxlen=self.max_lookback)
        self.volume_history = deque(maxlen=self.volume_trend_window)
        self.last_signal = 'HOLD'
        
    def reset(self) -> None:
        self.cumulative_pv = 0.0
        self.cumulative_volume = 0.0
        self.price_history.clear()
        self.volume_history.clear()
        self.last_signal = 'HOLD'
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        current_price = tick_data.price
        current_volume = tick_data.volume
        current_atr = tick_data.atr
        
        self.cumulative_pv += current_price * current_volume
        self.cumulative_volume += current_volume
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        
        if self.cumulative_volume == 0 or len(self.price_history) < 10:
            return None
            
        vwap = self.cumulative_pv / self.cumulative_volume
        
        deviation = abs(current_price - vwap)
        normalized_deviation = deviation / current_atr if current_atr > 0 else 0
        
        if len(self.volume_history) < self.volume_trend_window:
            return None
            
        recent_volumes = list(self.volume_history)
        volume_trend = (recent_volumes[-1] - recent_volumes[0]) / len(recent_volumes)
        volume_declining = volume_trend < 0
        
        if normalized_deviation > self.deviation_threshold and volume_declining:
            if current_price > vwap and self.last_signal != 'SELL':
                confidence = min(1.0, normalized_deviation / 4.0)
                self.last_signal = 'SELL'
                return Signal(
                    action='SELL',
                    confidence=confidence,
                    meta={
                        'vwap': vwap,
                        'deviation_atr': normalized_deviation,
                        'volume_trend': volume_trend,
                        'reversion_target': vwap
                    }
                )
            elif current_price < vwap and self.last_signal != 'BUY':
                confidence = min(1.0, normalized_deviation / 4.0)
                self.last_signal = 'BUY'
                return Signal(
                    action='BUY',
                    confidence=confidence,
                    meta={
                        'vwap': vwap,
                        'deviation_atr': normalized_deviation,
                        'volume_trend': volume_trend,
                        'reversion_target': vwap
                    }
                )
                
        return None