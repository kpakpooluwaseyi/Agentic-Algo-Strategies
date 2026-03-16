class BidAskMomentumStrategy(BaseStrategy):
    def __init__(self, spread_lookback: int = 10, momentum_period: int = 3):
        self.spread_lookback = spread_lookback
        self.momentum_period = momentum_period
        self.spreads = deque(maxlen=spread_lookback)
        self.mid_prices = deque(maxlen=momentum_period + 1)
        
    def reset(self) -> None:
        self.spreads.clear()
        self.mid_prices.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        bid = tick_data.bid
        ask = tick_data.ask
        price = tick_data.price
        
        mid = (bid + ask) / 2.0
        spread = ask - bid
        
        self.mid_prices.append(mid)
        self.spreads.append(spread)
        
        if len(self.spreads) < self.spread_lookback:
            return None
            
        avg_spread = statistics.mean(self.spreads)
        
        if len(self.mid_prices) < self.momentum_period + 1:
            return None
            
        past_mid = list(self.mid_prices)[0]
        current_mid = list(self.mid_prices)[-1]
        mid_change = (current_mid - past_mid) / past_mid if past_mid != 0 else 0
        
        spread_tightness = spread / avg_spread if avg_spread > 0 else 1.0
        
        if spread_tightness < 0.8:
            if mid_change > 0.001:
                confidence = min(1.0, abs(mid_change) * 100)
                return Signal(
                    action='BUY',
                    confidence=confidence,
                    meta={'spread_tightness': spread_tightness, 'mid_change': mid_change}
                )
            elif mid_change < -0.001:
                confidence = min(1.0, abs(mid_change) * 100)
                return Signal(
                    action='SELL',
                    confidence=confidence,
                    meta={'spread_tightness': spread_tightness, 'mid_change': mid_change}
                )
                
        return Signal(action='HOLD', confidence=0.0, meta={})