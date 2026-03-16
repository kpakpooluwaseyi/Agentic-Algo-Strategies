class MomentumVolumeStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.price_history = deque(maxlen=15)
        self.volume_history = deque(maxlen=15)
        
    def reset(self) -> None:
        self.price_history.clear()
        self.volume_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        timestamp = tick_data.timestamp
        
        current_signal = Signal(action='HOLD', confidence=0.0, meta={'timestamp': timestamp})
        
        if len(self.price_history) >= 15:
            old_price = self.price_history[0]
            price_change_pct = (price - old_price) / old_price if old_price != 0 else 0
            
            avg_volume = sum(self.volume_history) / len(self.volume_history)
            volume_surge = volume / avg_volume if avg_volume > 0 else 1.0
            
            if price_change_pct > 0.015 and volume_surge > 1.3:
                confidence = min(1.0, abs(price_change_pct) / 0.05)
                current_signal = Signal(
                    action='BUY',
                    confidence=confidence,
                    meta={'roc': price_change_pct, 'volume_surge': volume_surge, 'lookback': 15}
                )
            elif price_change_pct < -0.015 and volume_surge > 1.3:
                confidence = min(1.0, abs(price_change_pct) / 0.05)
                current_signal = Signal(
                    action='SELL',
                    confidence=confidence,
                    meta={'roc': price_change_pct, 'volume_surge': volume_surge, 'lookback': 15}
                )
            else:
                current_signal = Signal(
                    action='HOLD',
                    confidence=0.0,
                    meta={'roc': price_change_pct, 'volume_surge': volume_surge}
                )
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        return current_signal