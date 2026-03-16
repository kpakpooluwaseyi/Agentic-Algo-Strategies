class BidAskReversalStrategy(BaseStrategy):
    def __init__(self, lookback: int = 10, imbalance_threshold: float = 0.2):
        self.lookback = lookback
        self.imbalance_threshold = imbalance_threshold
        self.prices: List[float] = []
        self.spreads: List[float] = []
        
    def reset(self) -> None:
        self.prices = []
        self.spreads = []
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        atr = tick_data.atr
        
        if ask <= bid:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_pct = spread / mid if mid > 0 else 0
        
        self.prices.append(price)
        self.spreads.append(spread_pct)
        
        if len(self.prices) > self.lookback:
            self.prices.pop(0)
            self.spreads.pop(0)
            
        if len(self.prices) < self.lookback:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        avg_spread = sum(self.spreads) / len(self.spreads)
        price_change = (self.prices[-1] - self.prices[0]) / (atr + 1e-9)
        
        distance_to_bid = (price - bid) / (ask - bid) if (ask - bid) > 0 else 0.5
        
        if spread_pct < avg_spread * 0.9:
            if distance_to_bid < self.imbalance_threshold and price_change > 0.5:
                confidence = min(1.0, 0.5 + abs(price_change) * 0.1)
                return Signal(action='BUY', confidence=confidence, meta={
                    'distance_to_bid': distance_to_bid,
                    'spread_compression': spread_pct / avg_spread if avg_spread > 0 else 1.0,
                    'momentum': price_change
                })
            elif distance_to_bid > (1 - self.imbalance_threshold) and price_change < -0.5:
                confidence = min(1.0, 0.5 + abs(price_change) * 0.1)
                return Signal(action='SELL', confidence=confidence, meta={
                    'distance_to_ask': 1 - distance_to_bid,
                    'spread_compression': spread_pct / avg_spread if avg_spread > 0 else 1.0,
                    'momentum': price_change
                })
                
        return Signal(action='HOLD', confidence=0.0, meta={})