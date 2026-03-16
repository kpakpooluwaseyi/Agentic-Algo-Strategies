class VWAPSpreadReversion(BaseStrategy):
    def __init__(self, vwap_period: int = 30, deviation_pct: float = 0.003):
        self.vwap_period = vwap_period
        self.deviation_pct = deviation_pct
        self.price_volume_products: deque = deque(maxlen=vwap_period)
        self.volumes: deque = deque(maxlen=vwap_period)
        self.prev_mid: Optional[float] = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        bid = tick_data.bid
        ask = tick_data.ask
        
        mid = (bid + ask) / 2
        spread = ask - bid
        
        self.price_volume_products.append(price * volume)
        self.volumes.append(volume)
        
        if len(self.volumes) < self.vwap_period or sum(self.volumes) == 0:
            self.prev_mid = mid
            return None
            
        vwap = sum(self.price_volume_products) / sum(self.volumes)
        deviation = (price - vwap) / vwap
        spread_pct = spread / mid if mid > 0 else 0
        
        signal = None
        
        if abs(deviation) > self.deviation_pct:
            if deviation < 0 and price <= bid + (spread * 0.3):
                confidence = min(abs(deviation) / self.deviation_pct * 0.4 + 0.5, 0.9)
                signal = Signal(
                    action='BUY',
                    confidence=confidence,
                    meta={'vwap': vwap, 'deviation': deviation, 'spread_pct': spread_pct, 'urgency': 'aggressive_bid'}
                )
            elif deviation > 0 and price >= ask - (spread * 0.3):
                confidence = min(abs(deviation) / self.deviation_pct * 0.4 + 0.5, 0.9)
                signal = Signal(
                    action='SELL',
                    confidence=confidence,
                    meta={'vwap': vwap, 'deviation': deviation, 'spread_pct': spread_pct, 'urgency': 'aggressive_ask'}
                )
            else:
                signal = Signal(action='HOLD', confidence=0.2, meta={'vwap': vwap, 'deviation': deviation})
        else:
            signal = Signal(action='HOLD', confidence=0.5, meta={'vwap': vwap, 'status': 'neutral_zone'})
        
        self.prev_mid = mid
        return signal
    
    def reset(self) -> None:
        self.price_volume_products.clear()
        self.volumes.clear()
        self.prev_mid = None