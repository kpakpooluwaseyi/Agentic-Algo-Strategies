class BidAskReversionStrategy(BaseStrategy):
    def __init__(self):
        self.mid_prices = deque(maxlen=30)
        self.spread_history = deque(maxlen=30)
        self.timestamps = deque(maxlen=2)
        
    def reset(self) -> None:
        self.mid_prices.clear()
        self.spread_history.clear()
        self.timestamps.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        mid_price = (tick_data.bid + tick_data.ask) / 2.0
        spread = tick_data.ask - tick_data.bid
        
        if mid_price == 0:
            return None
            
        spread_pct = spread / mid_price
        
        self.mid_prices.append(mid_price)
        self.spread_history.append(spread_pct)
        self.timestamps.append(tick_data.timestamp)
        
        if len(self.mid_prices) < 20:
            return None
            
        mean_price = sum(self.mid_prices) / len(self.mid_prices)
        mean_spread = sum(self.spread_history) / len(self.spread_history)
        
        if mean_price == 0 or mean_spread == 0:
            return None
            
        deviation = (mid_price - mean_price) / mean_price
        spread_factor = spread_pct / mean_spread
        
        if abs(deviation) > 0.002 and spread_factor > 1.5:
            confidence = min(0.9, 0.5 + abs(deviation) * 100 + (spread_factor - 1) * 0.2)
            if deviation > 0:
                return Signal(action='SELL', confidence=confidence,
                            meta={'mean_deviation': deviation, 'spread_expansion': spread_factor})
            else:
                return Signal(action='BUY', confidence=confidence,
                            meta={'mean_deviation': deviation, 'spread_expansion': spread_factor})
                            
        return None