"""
strategy_997045cf76a0: vumanchu_scalping
"""
from .base_strategy import BaseStrategy
from src.config import MONITORED_TOKENS
import pandas as pd
import talib
from src import nice_funcs as n
from src.indicators.vumanchu import cipher_b

class VumanchuScalping(BaseStrategy):
    def __init__(self):
        super().__init__("VumanchuScalping")
        self.ema_fast_period = 50
        self.ema_slow_period = 200

    def generate_signals(self) -> dict:
        """
        Generate trading signals based on VuManchu Scalping logic.
        """
        for token in MONITORED_TOKENS:
            # Get data - using 10 days to ensure enough for 200 EMA
            data = n.get_data(token, days_back=10, timeframe='15m')
            if data is None or data.empty or len(data) < self.ema_slow_period:
                continue

            # Add indicators
            data['ema_fast'] = talib.EMA(data['close'], timeperiod=self.ema_fast_period)
            data['ema_slow'] = talib.EMA(data['close'], timeperiod=self.ema_slow_period)
            # vumanchu.py expects capitalized columns
            data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            data = cipher_b(data)

            # Get latest values
            latest = data.iloc[-1]
            prev = data.iloc[-2]

            # Detect trend via EMA crossover state
            is_bullish_trend = latest['ema_fast'] > latest['ema_slow']
            is_bearish_trend = latest['ema_fast'] < latest['ema_slow']

            direction = 'NEUTRAL'
            # Long entry signal
            if is_bullish_trend and latest['buy_signal']:
                direction = 'BUY'

            # Short entry signal
            elif is_bearish_trend and latest['sell_signal']:
                direction = 'SELL'

            if direction != 'NEUTRAL':
                return {
                    'token': token,
                    'signal': 1.0,
                    'direction': direction,
                    'metadata': {
                        'strategy_type': 'vumanchu_scalping',
                        'ema_fast': latest['ema_fast'],
                        'ema_slow': latest['ema_slow'],
                        'current_price': latest['Close']
                    }
                }
        return None
