class BidAskMeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.mid_prices = []
        self.lookback = 10
        
    def reset(self) -> None:
        self.mid_prices = []
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        bid = tick_data.bid
        ask = tick_data.ask
        price = tick_data.price
        atr = tick_data.atr
        
        mid = (bid + ask) / 2.0
        spread = ask - bid
        
        if len(self.mid_prices) < self.lookback:
            self.mid_prices.append(mid)
            return None
            
        self.mid_prices.append(mid)
        if len(self.mid_prices) > self.lookback:
            self.mid_prices.pop(0)
            
        sma = sum(self.mid_prices) / len(self.mid_prices)
        deviation = price - sma
        deviation_pct = deviation / sma if sma != 0 else 0.0
        
        # Filter: only trade when spread is tight relative to volatility
        if atr > 0 and spread > 0.4 * atr:
            return Signal(action='HOLD', confidence=0.0, meta={'filter': 'wide_spread', 'spread': spread})
            
        threshold = 0.6 * atr if atr > 0 else price * 0.002
        
        if deviation > threshold:
            confidence = min(1.0, (deviation / (threshold + 1e-9)) * 0.4)
            return Signal(action='SELL', confidence=confidence, meta={'deviation': deviation, 'sma': sma, 'signal': 'overbought'})
        elif deviation < -threshold:
            confidence = min(1.0, (abs(deviation) / (threshold + 1e-9)) * 0.4)
            return Signal(action='BUY', confidence=confidence, meta={'deviation': deviation, 'sma': sma, 'signal': 'oversold'})
        return Signal(action='HOLD', confidence=0.0, meta={'deviation': deviation, 'sma': sma})