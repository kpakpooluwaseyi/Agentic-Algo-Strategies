class BidAskImbalanceMeanReversionStrategy(BaseStrategy):
    def __init__(self):
        self.mid_prices: Deque[float] = deque(maxlen=50)
        self.entry_threshold: float = 0.4
        
    def reset(self) -> None:
        self.mid_prices.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        mid_price = (tick_data.bid + tick_data.ask) / 2.0
        self.mid_prices.append(mid_price)
        
        if len(self.mid_prices) < 50:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'warming_up'})
            
        sma_mid = statistics.mean(self.mid_prices)
        spread = tick_data.ask - tick_data.bid
        spread_pct = (spread / tick_data.price) if tick_data.price != 0 else float('inf')
        
        deviation = tick_data.price - sma_mid
        deviation_in_atr = deviation / tick_data.atr if tick_data.atr != 0 else 0
        
        meta = {
            'sma_mid': sma_mid,
            'deviation_atr': deviation_in_atr,
            'spread_pct': spread_pct,
            'mid_price': mid_price
        }
        
        if spread_pct < 0.0005:
            if deviation_in_atr < -self.entry_threshold:
                return Signal(action='BUY', confidence=min(0.9, abs(deviation_in_atr) / 2), meta=meta)
            elif deviation_in_atr > self.entry_threshold:
                return Signal(action='SELL', confidence=min(0.9, abs(deviation_in_atr) / 2), meta=meta)
        
        return Signal(action='HOLD', confidence=0.5, meta=meta)