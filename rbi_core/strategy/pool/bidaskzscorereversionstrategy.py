class BidAskZScoreReversionStrategy(BaseStrategy):
    def __init__(self, lookback: int = 15, zscore_threshold: float = 2.0, spread_threshold: float = 0.5):
        super().__init__()
        self.lookback = lookback
        self.zscore_threshold = zscore_threshold
        self.spread_threshold = spread_threshold
        self.price_history = deque(maxlen=lookback)
        self.spread_history = deque(maxlen=lookback)
        
    def reset(self) -> None:
        self.price_history.clear()
        self.spread_history.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        mid_price = (tick_data.bid + tick_data.ask) / 2
        spread = tick_data.ask - tick_data.bid
        atr = tick_data.atr
        
        if len(self.price_history) < self.lookback:
            self.price_history.append(mid_price)
            if atr > 0:
                self.spread_history.append(spread / atr)
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        mean_price = sum(self.price_history) / len(self.price_history)
        variance = sum((p - mean_price) ** 2 for p in self.price_history) / len(self.price_history)
        std_dev = math.sqrt(variance)
        
        z_score = (mid_price - mean_price) / std_dev if std_dev > 0 else 0
        
        self.price_history.append(mid_price)
        current_norm_spread = spread / atr if atr > 0 else 0
        self.spread_history.append(current_norm_spread)
        
        avg_norm_spread = sum(self.spread_history) / len(self.spread_history)
        
        if z_score > self.zscore_threshold and current_norm_spread > avg_norm_spread * (1 + self.spread_threshold):
            confidence = min(0.9, 0.5 + (z_score - self.zscore_threshold) * 0.1)
            return Signal(action='SELL', confidence=confidence, meta={
                'zscore': z_score,
                'spread_ratio': current_norm_spread / avg_norm_spread if avg_norm_spread > 0 else 0,
                'mean_price': mean_price
            })
        elif z_score < -self.zscore_threshold and current_norm_spread > avg_norm_spread * (1 + self.spread_threshold):
            confidence = min(0.9, 0.5 + (abs(z_score) - self.zscore_threshold) * 0.1)
            return Signal(action='BUY', confidence=confidence, meta={
                'zscore': z_score,
                'spread_ratio': current_norm_spread / avg_norm_spread if avg_norm_spread > 0 else 0,
                'mean_price': mean_price
            })
        else:
            return Signal(action='HOLD', confidence=0.0, meta={
                'zscore': z_score,
                'avg_norm_spread': avg_norm_spread
            })