class MicrostructureScalpingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.spread_history = deque(maxlen=50)
        self.mid_price_history = deque(maxlen=10)
        self.tick_count = 0
        self.last_signal_time = 0
        
    def reset(self) -> None:
        self.spread_history.clear()
        self.mid_price_history.clear()
        self.tick_count = 0
        self.last_signal_time = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        bid = tick_data.bid
        ask = tick_data.ask
        price = tick_data.price
        timestamp = tick_data.timestamp
        
        self.tick_count += 1
        
        # Calculate microstructure metrics
        mid_price = (bid + ask) / 2
        spread = ask - bid
        relative_spread = spread / mid_price if mid_price > 0 else 0
        
        self.spread_history.append(spread)
        self.mid_price_history.append(mid_price)
        
        # Minimum data requirement
        if len(self.spread_history) < 10:
            return Signal(action='HOLD', confidence=0.0, meta={'phase': 'initialization'})
        
        # Dynamic spread statistics
        avg_spread = sum(self.spread_history) / len(self.spread_history)
        spread_std = statistics.stdev(self.spread_history) if len(self.spread_history) > 1 else 0
        
        # Price location within spread (0 = at bid, 1 = at ask)
        if spread > 0:
            location = (price - bid) / spread
        else:
            location = 0.5
            
        # Short-term momentum of mid price
        if len(self.mid_price_history) >= 3:
            recent_mid = list(self.mid_price_history)[-3:]
            mid_slope = (recent_mid[-1] - recent_mid[0]) / 2 if len(recent_mid) == 3 else 0
        else:
            mid_slope = 0
        
        # Scalping logic: tight spread + price pressure
        spread_tight = spread < (avg_spread - 0.5 * spread_std)
        
        # Buy signal: price near bid with upward mid-price momentum
        if location < 0.3 and mid_slope > 0 and spread_tight:
            confidence = min(1.0, (0.3 - location) * 2 + abs(mid_slope) * 10)
            return Signal(action='BUY', confidence=confidence, meta={
                'strategy': 'microstructure_scalp',
                'location_in_spread': location,
                'mid_slope': mid_slope,
                'spread_condition': 'tight',
                'spread_bps': relative_spread * 10000
            })
            
        # Sell signal: price near ask with downward mid-price momentum  
        elif location > 0.7 and mid_slope < 0 and spread_tight:
            confidence = min(1.0, (location - 0.7) * 2 + abs(mid_slope) * 10)
            return Signal(action='SELL', confidence=confidence, meta={
                'strategy': 'microstructure_scalp',
                'location_in_spread': location,
                'mid_slope': mid_slope,
                'spread_condition': 'tight',
                'spread_bps': relative_spread * 10000
            })
            
        else:
            return Signal(action='HOLD', confidence=0.0, meta={
                'strategy': 'microstructure_scalp',
                'spread_percentile': 'elevated' if spread > avg_spread else 'normal',
                'location': location,
                'slope': mid_slope
            })