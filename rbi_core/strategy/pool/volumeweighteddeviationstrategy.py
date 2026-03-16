class VolumeWeightedDeviationStrategy(BaseStrategy):
    def __init__(self, lookback: int = 30, deviation_threshold: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.deviation_threshold = deviation_threshold
        self.prices: Deque[float] = deque(maxlen=lookback)
        self.volumes: Deque[float] = deque(maxlen=lookback)
        self.atrs: Deque[float] = deque(maxlen=lookback)
        
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.atrs.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        self.prices.append(price)
        self.volumes.append(volume)
        self.atrs.append(atr)
        
        if len(self.prices) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'accumulating_data'})
            
        total_pv = sum(p * v for p, v in zip(self.prices, self.volumes))
        total_volume = sum(self.volumes)
        vwap = total_pv / total_volume if total_volume > 0 else price
        
        avg_volume = mean(self.volumes)
        avg_atr = mean(self.atrs)
        
        deviation = price - vwap
        normalized_deviation = deviation / avg_atr if avg_atr > 0 else 0
        volume_confirm = volume > (avg_volume * 1.2)
        
        if normalized_deviation < -self.deviation_threshold and volume_confirm:
            confidence = min(1.0, abs(normalized_deviation) / (self.deviation_threshold * 3))
            return Signal(
                action='BUY',
                confidence=confidence,
                meta={
                    'vwap': vwap,
                    'deviation': normalized_deviation,
                    'volume_ratio': volume / avg_volume if avg_volume > 0 else 1.0,
                    'signal_type': 'mean_reversion_oversold'
                }
            )
        elif normalized_deviation > self.deviation_threshold and volume_confirm:
            confidence = min(1.0, normalized_deviation / (self.deviation_threshold * 3))
            return Signal(
                action='SELL',
                confidence=confidence,
                meta={
                    'vwap': vwap,
                    'deviation': normalized_deviation,
                    'volume_ratio': volume / avg_volume if avg_volume > 0 else 1.0,
                    'signal_type': 'mean_reversion_overbought'
                }
            )
        else:
            return Signal(action='HOLD', confidence=0.0, meta={'vwap': vwap, 'deviation': normalized_deviation})