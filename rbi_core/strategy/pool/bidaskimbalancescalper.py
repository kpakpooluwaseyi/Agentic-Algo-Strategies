class BidAskImbalanceScalper(BaseStrategy):
    def __init__(self, momentum_window: int = 8, spread_threshold: float = 0.0003):
        self.momentum_window = momentum_window
        self.spread_threshold = spread_threshold
        self.mid_prices: List[float] = []
        self.price_velocity = 0.0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        bid = tick_data.bid
        ask = tick_data.ask
        price = tick_data.price
        
        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid if mid > 0 else 0
        
        self.mid_prices.append(mid)
        if len(self.mid_prices) > self.momentum_window:
            self.mid_prices.pop(0)
            
        if len(self.mid_prices) < 3:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        if spread_pct > self.spread_threshold:
            return Signal(action='HOLD', confidence=0.0, meta={
                'spread_pct': spread_pct,
                'reason': 'excessive_spread'
            })
            
        short_ma = sum(self.mid_prices[-3:]) / 3.0
        long_ma = sum(self.mid_prices) / len(self.mid_prices)
        
        if len(self.mid_prices) >= 2:
            self.price_velocity = self.mid_prices[-1] - self.mid_prices[-2]
            
        trend_strength = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
        
        if self.price_velocity > 0 and trend_strength > 0.0001:
            confidence = min(0.9, abs(trend_strength) * 1000 + abs(self.price_velocity) * 10)
            return Signal(action='BUY', confidence=confidence, meta={
                'mid_price': mid,
                'spread_pct': spread_pct,
                'velocity': self.price_velocity,
                'trend_strength': trend_strength,
                'reason': 'micro_momentum_up'
            })
        elif self.price_velocity < 0 and trend_strength < -0.0001:
            confidence = min(0.9, abs(trend_strength) * 1000 + abs(self.price_velocity) * 10)
            return Signal(action='SELL', confidence=confidence, meta={
                'mid_price': mid,
                'spread_pct': spread_pct,
                'velocity': self.price_velocity,
                'trend_strength': trend_strength,
                'reason': 'micro_momentum_down'
            })
        else:
            return Signal(action='HOLD', confidence=0.0, meta={
                'velocity': self.price_velocity,
                'trend_strength': trend_strength
            })
            
    def reset(self) -> None:
        self.mid_prices.clear()
        self.price_velocity = 0.0