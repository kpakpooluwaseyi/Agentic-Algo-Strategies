class MicrostructureSentimentStrategy(BaseStrategy):
    def __init__(self, window: int = 10):
        self.window = window
        self.trade_imbalance = deque(maxlen=window)
        self.prev_bid = None
        self.prev_ask = None
        self.vwap_num = 0.0
        self.vwap_denom = 0.0
        self.tick_count = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        volume = tick_data.volume
        
        mid = (bid + ask) / 2.0
        spread = ask - bid
        
        if spread <= 0 or tick_data.atr <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={'error': 'invalid_spread'})
        
        micro_price = (bid * tick_data.ask + ask * tick_data.bid) / (tick_data.bid + tick_data.ask) if (bid + ask) > 0 else mid
        
        if price >= ask:
            sentiment = 1.0
        elif price <= bid:
            sentiment = -1.0
        else:
            sentiment = (price - mid) / (spread / 2)
            
        self.trade_imbalance.append(sentiment * volume)
        self.vwap_num += price * volume
        self.vwap_denom += volume
        self.tick_count += 1
        
        if len(self.trade_imbalance) < self.window:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'warming_up'})
        
        avg_sentiment = sum(self.trade_imbalance) / self.window
        vwap = self.vwap_num / self.vwap_denom if self.vwap_denom > 0 else price
        
        price_deviation = (price - vwap) / tick_data.atr
        
        if avg_sentiment > 0.7 and price_deviation < 0.5:
            return Signal(
                action='BUY',
                confidence=min(1.0, abs(avg_sentiment) * 0.8 + (0.5 - price_deviation) * 0.4),
                meta={'sentiment': avg_sentiment, 'vwap_deviation': price_deviation}
            )
        elif avg_sentiment < -0.7 and price_deviation > -0.5:
            return Signal(
                action='SELL',
                confidence=min(1.0, abs(avg_sentiment) * 0.8 + (0.5 + price_deviation) * 0.4),
                meta={'sentiment': avg_sentiment, 'vwap_deviation': price_deviation}
            )
            
        return Signal(action='HOLD', confidence=abs(avg_sentiment) * 0.3, meta={'sentiment': avg_sentiment})
    
    def reset(self) -> None:
        self.trade_imbalance.clear()
        self.prev_bid = None
        self.prev_ask = None
        self.vwap_num = 0.0
        self.vwap_denom = 0.0
        self.tick_count = 0