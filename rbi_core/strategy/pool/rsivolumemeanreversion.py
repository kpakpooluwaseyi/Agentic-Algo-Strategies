class RsiVolumeMeanReversion(BaseStrategy):
    def __init__(self, period: int = 14, overbought: float = 70.0, oversold: float = 30.0):
        super().__init__()
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.prices = deque(maxlen=period + 1)
        self.volumes = deque(maxlen=period)
        self.prev_price = None
        
    def reset(self) -> None:
        self.prices.clear()
        self.volumes.clear()
        self.prev_price = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        if self.prev_price is None:
            self.prev_price = price
            return None
            
        self.prices.append(price)
        
        if len(self.prices) < self.period + 1:
            self.prev_price = price
            return None
            
        # Calculate RSI
        gains = []
        losses = []
        prices_list = list(self.prices)
        
        for i in range(1, len(prices_list)):
            change = prices_list[i] - prices_list[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))
                
        avg_gain = sum(gains) / len(gains) if gains else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
        # Volume analysis
        self.volumes.append(volume)
        avg_volume = sum(self.volumes) / len(self.volumes) if self.volumes else volume
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        action = 'HOLD'
        confidence = 0.0
        meta = {
            'rsi': rsi,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,