class RSIOverboughtOversoldStrategy(BaseStrategy):
    def __init__(self):
        self.prices = []
        self.rsi_period = 14

    def calculate_rsi(self):
        if len(self.prices) < self.rsi_period:
            return None
        
        gains = 0
        losses = 0
        
        for i in range(1, self.rsi_period + 1):
            change = self.prices[-i] - self.prices[-(i + 1)]
            if change > 0:
                gains += change
            else:
                losses -= change
        
        avg_gain = gains / self.rsi_period
        avg_loss = losses / self.rsi_period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def on_tick(self, tick_data):
        self.prices.append(tick_data.price)
        if len(self.prices) > self.rsi_period:
            self.prices.pop(0)

        rsi = self.calculate_rsi()
        if rsi is not None:
            if rsi < 30:
                return Signal(action='BUY', confidence=1.0, meta={'rsi': rsi})
            elif rsi > 70:
                return Signal(action='SELL', confidence=1.0, meta={'rsi': rsi})

        return Signal(action='HOLD', confidence=0.5, meta={})

    def reset(self):
        self.prices.clear()