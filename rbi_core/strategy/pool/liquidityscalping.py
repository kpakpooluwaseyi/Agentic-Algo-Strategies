from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque

class LiquidityScalping(BaseStrategy):
    def __init__(self, spread_window: int = 20, momentum_window: int = 5, spread_threshold: float = 0.8):
        self.spread_window = spread_window
        self.momentum_window = momentum_window
        self.spread_threshold = spread_threshold
        self.spreads = deque(maxlen=spread_window)
        self.prices = deque(maxlen=momentum_window)
        self.last_signal_time = 0
        
    def reset(self) -> None:
        self.spreads.clear()
        self.prices.clear()
        self.last_signal_time = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        bid = tick_data.bid
        ask = tick_data.ask
        timestamp = tick_data.timestamp
        
        if timestamp == self.last_signal_time:
            return None
            
        spread = ask - bid
        if spread <= 0:
            return None
            
        self.spreads.append(spread)
        self.prices.append(price)
        
        if len(self.spreads) < self.spread_window or len(self.prices) < self.momentum_window:
            return None
            
        avg_spread = sum(self.spreads) / len(self.spreads)
        
        if spread > avg_spread * self.spread_threshold:
            return None
            
        momentum = (price - self.prices[0]) / self.prices[0] if self.prices[0] > 0 else 0
        
        if abs(momentum) < 0.0005:
            return None
            
        self.last_signal_time = timestamp
        
        if momentum > 0:
            return Signal(
                action='BUY',
                confidence=min(0.9, 0.5 + abs(momentum) * 100),
                meta={'spread': spread, 'avg_spread': avg_spread, 'momentum': momentum, 'liquidity': 'tight'}
            )
        else:
            return Signal(
                action='SELL',
                confidence=min(0.9, 0.5 + abs(momentum) * 100),
                meta={'spread': spread, 'avg_spread': avg_spread, 'momentum': momentum, 'liquidity': 'tight'}
            )</think>from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque
import statistics

class ATRChannelBreakout(BaseStrategy):
    def __init__(self, lookback: int = 20, atr_multiplier: float = 2.0):
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier
        self.prices = deque(maxlen=lookback)
        self.position = 0
        
    def reset(self) -> None:
        self.prices.clear()
        self.position = 0
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        atr = tick_data.atr
        
        self.prices.append(price)
        
        if len(self.prices) < self.lookback:
            return None
            
        sma = statistics.mean(self.prices)
        upper_band = sma + (self.atr_multiplier * atr)
        lower_band = sma - (self.atr_multiplier * atr)
        
        if price > upper_band and self.position <= 0:
            self.position = 1
            return Signal(
                action='BUY',
                confidence=min(0.95, (price - upper_band) / (atr * 0.5) + 0.6),
                meta={'sma': sma, 'upper_band': upper_band, 'atr': atr}
            )
        elif price < lower_band and self.position >= 0:
            self.position = -1
            return Signal(
                action='SELL',
                confidence=min(0.95, (lower_band - price) / (atr * 0.5) + 0.6),
                meta={'sma': sma, 'lower_band': lower_band, 'atr': atr}
            )
            
        return None