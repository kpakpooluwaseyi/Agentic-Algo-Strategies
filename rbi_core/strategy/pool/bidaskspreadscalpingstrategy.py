class BidAskSpreadScalpingStrategy(BaseStrategy):
    def __init__(self, spread_threshold_percent: float = 0.1, price_history: int = 10):
        self.spread_threshold_percent = spread_threshold_percent
        self.price_history = price_history
        self.mid_prices = deque(maxlen=price_history)
        self.bid_ask_ratios = deque(maxlen=price_history)
        
    def reset(self) -> None:
        self.mid_prices.clear()
        self.bid_ask_ratios.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        bid = tick_data.bid
        ask = tick_data.ask
        price = tick_data.price
        
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_pct = (spread / mid * 100) if mid > 0 else 0
        
        if len(self.mid_prices) < self.price_history:
            self.mid_prices.append(mid)
            self.bid_ask_ratios.append(bid / ask if ask > 0 else 1.0)
            return Signal(action='HOLD', confidence=0.0, meta={'buffering': len(self.mid_prices)})
        
        avg_mid = statistics.mean(self.mid_prices)
        std_mid = statistics.stdev(self.mid_prices) if len(self.mid_prices) > 1 else 0
        
        self.mid_prices.append(mid)
        self.bid_ask_ratios.append(bid / ask if ask > 0 else 1.0)
        
        avg_ratio = statistics.mean(self.bid_ask_ratios)
        
        deviation = (mid - avg_mid) / avg_mid if avg_mid != 0 else 0
        
        if spread_pct < self.spread_threshold_percent:
            if deviation > 0.0005 and price > avg_mid + std_mid:
                confidence = min(1.0, abs(deviation) * 1000 + (1 - spread_pct/self.spread_threshold_percent) * 0.5)
                return Signal(action='SELL', confidence=confidence, meta={'deviation': deviation, 'spread_pct': spread_pct, 'imbalance': avg_ratio})
            elif deviation < -0.0005 and price < avg_mid - std_mid:
                confidence = min(1.0, abs(deviation) * 1000 + (1 - spread_pct/self.spread_threshold_percent) * 0.5)
                return Signal(action='BUY', confidence=confidence, meta={'deviation': deviation, 'spread_pct': spread_pct, 'imbalance': avg_ratio})
        
        return Signal(action='HOLD', confidence=0.0, meta={'spread_pct': spread_pct, 'deviation': deviation, 'mid': mid})