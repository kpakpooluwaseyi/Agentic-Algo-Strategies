class VolumeSpreadMomentumStrategy(BaseStrategy):
    def __init__(self, lookback: int = 10, volume_threshold: float = 1.5):
        self.lookback = lookback
        self.volume_threshold = volume_threshold
        self.prices: Deque[float] = deque(maxlen=lookback)
        self.volumes: Deque[float] = deque(maxlen=lookback)
        self.last_signal: str = 'HOLD'
    
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.last_signal = 'HOLD'
    
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'initializing'})
        
        avg_volume = sum(self.volumes) / len(self.volumes)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        start_price = self.prices[0]
        momentum = ((price - start_price) / start_price * 100) if start_price != 0 else 0
        
        spread = ask - bid
        mid = (ask + bid) / 2
        spread_pct = (spread / mid * 100) if mid > 0 else 0
        
        if spread > 0:
            position_in_spread = (price - bid) / spread
        else:
            position_in_spread = 0.5
        
        if volume_ratio > self.volume_threshold:
            if momentum > 0.5 and position_in_spread > 0.7:
                confidence = min(0.5 + (volume_ratio - 1.0) * 0.2 + momentum * 0.05, 1.0)
                if self.last_signal != 'BUY':
                    self.last_signal = 'BUY'
                    return Signal(
                        action='BUY',
                        confidence=confidence,
                        meta={
                            'volume_ratio': round(volume_ratio, 2),
                            'momentum_pct': round(momentum, 2),
                            'spread_pressure': round(position_in_spread, 2),
                            'spread_pct': round(spread_pct, 4)
                        }
                    )
            elif momentum < -0.5 and position_in_spread < 0.3:
                confidence = min(0.5 + (volume_ratio - 1.0) * 0.2 + abs(momentum) * 0.05, 1.0)
                if self.last_signal != 'SELL':
                    self.last_signal = 'SELL'
                    return Signal(
                        action='SELL',
                        confidence=confidence,
                        meta={
                            'volume_ratio': round(volume_ratio, 2),
                            'momentum_pct': round(momentum, 2),
                            'spread_pressure': round(position_in_spread, 2),
                            'spread_pct': round(spread_pct, 4)
                        }
                    )
        
        return Signal(
            action='HOLD',
            confidence=0.0,
            meta={
                'volume_ratio': round(volume_ratio, 2),
                'momentum_pct': round(momentum, 2),
                'spread_pressure': round(position_in_spread, 2)
            }
        )