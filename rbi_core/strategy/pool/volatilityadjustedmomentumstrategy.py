class VolatilityAdjustedMomentumStrategy(BaseStrategy):
    def __init__(self, momentum_period: int = 10, atr_threshold_multiplier: float = 0.8):
        super().__init__()
        self.momentum_period = momentum_period
        self.atr_threshold_multiplier = atr_threshold_multiplier
        self.prices = deque(maxlen=momentum_period + 1)
        self.volumes = deque(maxlen=momentum_period)
        self.atrs = deque(maxlen=momentum_period)
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        self.prices.append(tick_data.price)
        self.volumes.append(tick_data.volume)
        self.atrs.append(tick_data.atr)
        
        if len(self.prices) < self.momentum_period + 1:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        if tick_data.atr == 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        old_price = self.prices[0]
        price_change = tick_data.price - old_price
        normalized_momentum = price_change / tick_data.atr
        
        avg_volume = sum(self.volumes) / len(self.volumes)
        volume_participation = tick_data.volume / avg_volume if avg_volume > 0 else 1.0
        
        avg_atr = sum(self.atrs) / len(self.atrs)
        low_volatility_regime = tick_data.atr < avg_atr * self.atr_threshold_multiplier
        
        meta = {
            'normalized_momentum': normalized_momentum,
            'volume_participation': volume_participation,
            'low_volatility': low_volatility_regime
        }
        
        if normalized_momentum > 1.0 and low_volatility_regime and volume_participation > 1.0:
            confidence = min(0.85, 0.5 + (normalized_momentum / 4) + (volume_participation / 10))
            return Signal(action='BUY', confidence=confidence, meta={**meta, 'signal_type': 'momentum_in_low_vol'})
            
        elif normalized_momentum < -1.0 and low_volatility_regime and volume_participation > 1.0:
            confidence = min(0.85, 0.5 + (abs(normalized_momentum) / 4) + (volume_participation / 10))
            return Signal(action='SELL', confidence=confidence, meta={**meta, 'signal_type': 'momentum_in_low_vol'})
            
        return Signal(action='HOLD', confidence=0.0, meta=meta)
        
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.atrs.clear()