from rbi_core.strategy.base import BaseStrategy, Signal
import time

class MovingAverageCrossoverStrategy(BaseStrategy):
    def __init__(self):
        self.short_window = 5
        self.long_window = 20
        self.short_ma = []
        self.long_ma = []

    def on_tick(self, tick_data):
        self.short_ma.append(tick_data.price)
        self.long_ma.append(tick_data.price)

        if len(self.short_ma) > self.short_window:
            self.short_ma.pop(0)
        if len(self.long_ma) > self.long_window:
            self.long_ma.pop(0)

        if len(self.short_ma) == self.short_window and len(self.long_ma) == self.long_window:
            short_avg = sum(self.short_ma) / self.short_window
            long_avg = sum(self.long_ma) / self.long_window
            
            if short_avg > long_avg:
                return Signal(action='BUY', confidence=1.0, meta={'short_avg': short_avg, 'long_avg': long_avg})
            elif short_avg < long_avg:
                return Signal(action='SELL', confidence=1.0, meta={'short_avg': short_avg, 'long_avg': long_avg})
        
        return Signal(action='HOLD', confidence=0.5, meta={})

    def reset(self):
        self.short_ma.clear()
        self.long_ma.clear()