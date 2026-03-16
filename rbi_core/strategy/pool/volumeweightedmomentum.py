class VolumeWeightedMomentum(BaseStrategy):
    """
    A momentum strategy comparing Simple Moving Average (SMA) vs 
    Volume Weighted Moving Average (VWMA).
    Generates signals on crossovers, assuming volume confirms the trend.
    """
    def __init__(self, period: int = 15):
        super().__init__()
        self.period = period
        self.prices: deque = deque(maxlen=period)
        self.volumes: deque = deque(maxlen=period)
        self.trend_state: int = 0 # 1 for Bullish, -1 for Bearish, 0 for Flat

    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.trend_state = 0

    def on_tick(self, tick_data) -> Optional[Signal]:
        self.prices.append(tick_data.price)
        self.volumes.append(tick_data.volume)

        if len(self.prices) < self.period:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'warming_up'})

        # Calculate SMA
        sma = sum(self.prices) / self.period

        # Calculate VWMA
        # VWMA = Sum(Price * Volume) / Sum(Volume)
        total_volume = sum(self.volumes)
        if total_volume == 0:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'zero_volume'})
            
        weighted_sum = sum(p * v for p, v in zip(self.prices, self.volumes))
        vwma = weighted_sum / total_volume

        current_price = tick_data.price
        action = 'HOLD'
        confidence = 0.0
        meta = {'sma': sma, 'vwma': vwma}

        # Determine Trend
        # If VWMA > SMA, volume is supporting higher prices (Bullish)
        # If VWMA < SMA, volume is supporting lower prices (Bearish)
        new_trend = 0
        if vwma > sma:
            new_trend = 1
        elif vwma < sma:
            new_trend = -1
        
        # Detect Crossovers (State Change)
        if new_trend == 1 and self.trend_state != 1:
            # Bullish Crossover
            action = 'BUY'
            # Confidence based on the divergence between VWMA and SMA relative to price
            divergence = (vwma - sma) / current_price
            confidence = min(1.0, 0.6 + abs(divergence) * 100) 
            self.trend_state = 1
        elif new_trend == -1 and self.trend_state != -1:
            # Bearish Crossover
            action = 'SELL'
            divergence = (sma - vwma) / current_price
            confidence = min(1.0, 0.6 + abs(divergence) * 100)
            self.trend_state = -1
            
        return Signal(action=action, confidence=confidence, meta=meta)