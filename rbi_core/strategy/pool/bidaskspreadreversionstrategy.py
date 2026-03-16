class BidAskSpreadReversionStrategy(BaseStrategy):
    def __init__(self, window: int = 10, z_score_threshold: float = 1.5):
        super().__init__()
        self.window = window
        self.z_score_threshold = z_score_threshold
        self.mid_prices: Deque[float] = deque(maxlen=window)
        self.spreads: Deque[float] = deque(maxlen=window)
        
    def reset(self) -> None:
        self.mid_prices.clear()
        self.spreads.clear()
        
    def _calculate_std(self, values: Deque[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        bid = tick_data.bid
        ask = tick_data.ask
        price = tick_data.price
        
        if bid <= 0 or ask <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        mid_price = (bid + ask) / 2.0
        spread = ask - bid
        spread_pct = spread / mid_price if mid_price > 0 else 0
        
        if len(self.mid_prices) < self.window:
            self.mid_prices.append(mid_price)
            self.spreads.append(spread_pct)
            return Signal(action='HOLD', confidence=0.0, meta={})
        
        mean_mid = sum(self.mid_prices) / len(self.mid_prices)
        std_mid = self._calculate_std(self.mid_prices)
        mean_spread = sum(self.spreads) / len(self.spreads)
        
        z_score = (mid_price - mean_mid) / std_mid if std_mid > 0 else 0
        spread_expansion = spread_pct > mean_spread * 1.3
        
        self.mid_prices.append(mid_price)
        self.spreads.append(spread_pct)
        
        if z_score < -self.z_score_threshold and spread_expansion:
            confidence = min(0.85, abs(z_score) / 3.0)
            return Signal(action='BUY', confidence=confidence, meta={
                'z_score': z_score,
                'spread_pct': spread_pct,
                'mean_spread': mean_spread,
                'signal': 'liquidity_stress_reversion'
            })
        elif z_score > self.z_score_threshold and spread_expansion:
            confidence = min(0.85, abs(z_score) / 3.0)
            return Signal(action='SELL', confidence=confidence, meta={
                'z_score': z_score,
                'spread_pct': spread_pct,
                'mean_spread': mean_spread,
                'signal': 'liquidity_euphoria_reversion'
            })
        
        return Signal(action='HOLD', confidence=0.0, meta={})