class BollingerBandReversionStrategy(BaseStrategy):
    def __init__(self):
        self.prices = []
        self.window = 15
        
    def reset(self) -> None:
        self.prices.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        atr = tick_data.atr
        
        self.prices.append(price)
        if len(self.prices) > self.window:
            self.prices.pop(0)
            
        if len(self.prices) < self.window:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        n = len(self.prices)
        sma = sum(self.prices) / n
        
        variance = sum((p - sma) ** 2 for p in self.prices) / n
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
            
        spread_width = ask - bid
        if spread_width > 0:
            position_in_spread = (price - bid) / spread_width
        else:
            position_in_spread = 0.5
            
        upper_band = sma + 2 * std_dev
        lower_band = sma - 2 * std_dev
        
        if price < lower_band and position_in_spread < 0.35:
            deviation = (lower_band - price) / std_dev
            confidence = min(0.95, 0.4 + deviation * 0.3)
            return Signal(
                action='BUY',
                confidence=confidence,
                meta={'regime': 'oversold', 'sma': sma, 'z_score': -deviation}
            )
        elif price > upper_band and position_in_spread > 0.65:
            deviation = (price - upper_band) / std_dev
            confidence = min(0.95, 0.4 + deviation * 0.3)
            return Signal(
                action='SELL',
                confidence=confidence,
                meta={'regime': 'overbought', 'sma': sma, 'z_score': deviation}
            )
            
        return Signal(action='HOLD', confidence=0.0, meta={})