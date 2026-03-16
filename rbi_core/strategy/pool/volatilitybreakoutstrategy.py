from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class VolatilityBreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.price_window = deque(maxlen=20)
        self.volume_window = deque(maxlen=20)
        self.prev_signal = 'HOLD'
        self.last_trade_price = None
        
    def reset(self) -> None:
        self.price_window.clear()
        self.volume_window.clear()
        self.prev_signal = 'HOLD'
        self.last_trade_price = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        atr = tick_data.atr
        bid = tick_data.bid
        ask = tick_data.ask
        
        self.price_window.append(price)
        self.volume_window.append(volume)
        
        if len(self.price_window) < 20:
            return Signal(action='HOLD', confidence=0.0, meta={'reason': 'insufficient_data'})
        
        sma = sum(self.price_window) / len(self.price_window)
        avg_volume = sum(self.volume_window) / len(self.volume_window)
        
        spread = ask - bid
        mid = (bid + ask) / 2.0
        spread_pct = spread / mid if mid != 0 else 1.0
        
        upper_band = sma + (1.5 * atr)
        lower_band = sma - (1.5 * atr)
        
        volume_surge = volume > (avg_volume * 1.2)
        liquid_market = spread_pct < 0.0005
        
        action = 'HOLD'
        confidence = 0.0
        
        if price > upper_band and volume_surge and liquid_market:
            action = 'BUY'
            confidence = min(0.95, 0.5 + (price - upper_band) / (2 * atr)) if atr > 0 else 0.6
            if self.prev_signal == 'BUY' and self.last_trade_price:
                if abs(price - self.last_trade_price) / self.last_trade_price < 0.002:
                    action = 'HOLD'
                    confidence = 0.0
            if action == 'BUY':
                self.last_trade_price = price
                
        elif price < lower_band and volume_surge and liquid_market:
            action = 'SELL'
            confidence = min(0.95, 0.5 + (lower_band - price) / (2 * atr)) if atr > 0 else 0.6
            if self.prev_signal == 'SELL' and self.last_trade_price:
                if abs(price - self.last_trade_price) / self.last_trade_price < 0.002:
                    action = 'HOLD'
                    confidence = 0.0
            if action == 'SELL':
                self.last_trade_price = price
        
        self.prev_signal = action
        
        meta = {
            'sma': round(sma, 4),
            'atr': round(atr, 4),
            'upper_band': round(upper_band, 4),
            'lower_band': round(lower_band, 4),
            'volume_ratio': round(volume / avg_volume, 2) if avg_volume > 0 else 0,
            'spread_bps': round(spread_pct * 10000, 2)
        }
        
        return Signal(action=action, confidence=confidence, meta=meta)