class VPTDivergenceStrategy(BaseStrategy):
    def __init__(self, lookback: int = 15):
        self.lookback = lookback
        self.vpt = 0.0
        self.prices_history = deque(maxlen=lookback)
        self.vpt_history = deque(maxlen=lookback)
        self.prev_price = None
        
    def reset(self) -> None:
        self.vpt = 0.0
        self.prices_history.clear()
        self.vpt_history.clear()
        self.prev_price = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        
        if self.prev_price is None or self.prev_price == 0:
            self.prev_price = price
            self.prices_history.append(price)
            self.vpt_history.append(0.0)
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        price_change_pct = (price - self.prev_price) / self.prev_price
        self.vpt += volume * price_change_pct
        
        if len(self.prices_history) < self.lookback:
            self.prices_history.append(price)
            self.vpt_history.append(self.vpt)
            self.prev_price = price
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        price_trend = (price - self.prices_history[0]) / self.lookback
        vpt_trend = (self.vpt - self.vpt_history[0]) / self.lookback
        
        divergence_bull = vpt_trend > 0 and price_trend < 0
        divergence_bear = vpt_trend < 0 and price_trend > 0
        
        signal = Signal(action='HOLD', confidence=0.0, meta={'vpt': self.vpt, 'price_trend': price_trend})
        
        if divergence_bull and abs(price_change_pct) > atr * 0.01:
            confidence = min(0.85, 0.5 + math.atan(abs(vpt_trend)) / math.pi * 2)
            signal = Signal(action='BUY', confidence=confidence, meta={
                'divergence': 'bullish',
                'vpt_trend': vpt_trend,
                'price_trend': price_trend,
                'vpt_value': self.vpt
            })
        elif divergence_bear and abs(price_change_pct) > atr * 0.01:
            confidence = min(0.85, 0.5 + math.atan(abs(vpt_trend)) / math.pi * 2)
            signal = Signal(action='SELL', confidence=confidence, meta={
                'divergence': 'bearish',
                'vpt_trend': vpt_trend,
                'price_trend': price_trend,
                'vpt_value': self.vpt
            })
        
        self.prices_history.append(price)
        self.vpt_history.append(self.vpt)
        self.prev_price = price
        
        return signal