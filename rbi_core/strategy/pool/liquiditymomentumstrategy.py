class LiquidityMomentumStrategy(BaseStrategy):
    def __init__(self, spread_lookback: int = 20, momentum_periods: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.spread_lookback = spread_lookback
        self.momentum_periods = momentum_periods
        self.spreads: Deque[float] = deque(maxlen=spread_lookback)
        self.mid_prices: Deque[float] = deque(maxlen=momentum_periods * 2)
        self.timestamps: Deque[float] = deque(maxlen=2)
        
    def reset(self) -> None:
        self.spreads.clear()
        self.mid_prices.clear()
        self.timestamps.clear()
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        bid = tick_data.bid
        ask = tick_data.ask
        timestamp = tick_data.timestamp
        
        mid_price = (bid + ask) / 2
        spread = ask - bid
        
        self.spreads.append(spread)
        self.mid_prices.append(mid_price)
        self.timestamps.append(timestamp)
        
        if len(self.spreads) < self.spread_lookback or len(self.mid_prices) < self.momentum_periods * 2:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'warming_up'})
            
        avg_spread = mean(self.spreads)
        liquidity_improving = spread < (avg_spread * 0.85)
        
        past_price = list(self.mid_prices)[-self.momentum_periods - 1]
        recent_price = list(self.mid_prices)[-1]
        
        if past_price == 0:
            return Signal(action='HOLD', confidence=0.0, meta={'error': 'zero_price'})
            
        price_change_pct = (recent_price - past_price) / past_price
        
        if liquidity_improving:
            if price_change_pct > 0.0005:
                confidence = min(1.0, price_change_pct * 1000)
                return Signal(
                    action='BUY',
                    confidence=confidence,
                    meta={
                        'momentum': price_change_pct,
                        'spread_ratio': spread / avg_spread,
                        'liquidity': 'improving',
                        'mid_price': mid_price
                    }
                )
            elif price_change_pct < -0.0005:
                confidence = min(1.0, abs(price_change_pct) * 1000)
                return Signal(
                    action='SELL',
                    confidence=confidence,
                    meta={
                        'momentum': price_change_pct,
                        'spread_ratio': spread / avg_spread,
                        'liquidity': 'improving',
                        'mid_price': mid_price
                    }
                )
                
        return Signal(
            action='HOLD',
            confidence=0.0,
            meta={
                'spread_ratio': spread / avg_spread if avg_spread > 0 else 1.0,
                'momentum': price_change_pct
            }
        )