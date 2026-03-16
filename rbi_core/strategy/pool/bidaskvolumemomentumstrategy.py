class BidAskVolumeMomentumStrategy(BaseStrategy):
    def __init__(self, volume_lookback: int = 10, momentum_threshold: float = 0.001, spread_max_pct: float = 0.002):
        super().__init__()
        self.volume_lookback = volume_lookback
        self.momentum_threshold = momentum_threshold
        self.spread_max_pct = spread_max_pct
        self.volume_history = deque(maxlen=volume_lookback)
        self.prev_price = None
        
    def reset(self) -> None:
        self.volume_history.clear()
        self.prev_price = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        if self.prev_price is None:
            self.prev_price = price
            self.volume_history.append(volume)
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'priming'})
        
        momentum = (price - self.prev_price) / self.prev_price if self.prev_price != 0 else 0
        self.prev_price = price
        self.volume_history.append(volume)
        
        if len(self.volume_history) < self.volume_lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'initializing'})
        
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        volume_surge = volume / avg_volume if avg_volume > 0 else 1.0
        
        mid_price = (bid + ask) / 2
        spread_pct = (ask - bid) / mid_price if mid_price > 0 else float('inf')
        
        if spread_pct > self.spread_max_pct:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'excessive_spread', 'spread_pct': spread_pct})
        
        if momentum > self.momentum_threshold and volume_surge > 1.2:
            confidence = min((momentum / self.momentum_threshold) * 0.5 + (volume_surge - 1.0) * 0.25, 1.0)
            return Signal(
                action='BUY',
                confidence=confidence,
                meta={
                    'strategy': 'bidask_volume_momentum',
                    'momentum': momentum,
                    'volume_surge': volume_surge,
                    'spread_pct': spread_pct
                }
            )
        elif momentum < -self.momentum_threshold and volume_surge > 1.2:
            confidence = min((abs(momentum) / self.momentum_threshold) * 0.5 + (volume_surge - 1.0) * 0.25, 1.0)
            return Signal(
                action='SELL',
                confidence=confidence,
                meta={
                    'strategy': 'bidask_volume_momentum',
                    'momentum': momentum,
                    'volume_surge': volume_surge,
                    'spread_pct': spread_pct
                }
            )
        
        return Signal(action='HOLD', confidence=0.0, meta={'momentum': momentum, 'volume_surge': volume_surge})