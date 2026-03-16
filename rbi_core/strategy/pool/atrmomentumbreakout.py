class ATRMomentumBreakout(BaseStrategy):
    def __init__(self):
        self.lookback = 5
        self.volume_window = 20
        self.atr_mult = 1.5
        self.price_history = deque(maxlen=self.lookback)
        self.volume_history = deque(maxlen=self.volume_window)
        
    def reset(self) -> None:
        self.price_history.clear()
        self.volume_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        if len(self.price_history) < self.lookback:
            self.price_history.append(price)
            self.volume_history.append(volume)
            return None
            
        price_n_periods_ago = self.price_history[0]
        price_change = price - price_n_periods_ago
        atr_threshold = (atr * self.atr_mult) if atr else 0.0
        
        avg_volume = sum(self.volume_history) / len(self.volume_history) if self.volume_history else 0.0
        
        action = 'HOLD'
        confidence = 0.0
        meta = {'price_change': price_change, 'atr_threshold': atr_threshold}
        
        if abs(price_change) > atr_threshold and volume > avg_volume * 1.2 and atr_threshold > 0:
            if price_change > 0:
                action = 'BUY'
                confidence = min(1.0, 0.5 + (abs(price_change) - atr_threshold) / (atr_threshold * 2))
            else:
                action = 'SELL'
                confidence = min(1.0, 0.5 + (abs(price_change) -