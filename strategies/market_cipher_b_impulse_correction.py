import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from src.indicators.market_cipher_b import market_cipher_b
import pandas_ta as ta

class MarketCipherBImpulseCorrection(Strategy):
    # Default parameters
    ema_period = 200
    mfi_overbought = 80
    mfi_oversold = 20
    tp_pct = 0.015  # 1.5%
    sl_pct = 0.0075 # 0.75%

    def init(self):
        # Higher timeframe trend
        self.ema = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_period)

        # Market Cipher B indicators
        self.wt1, self.wt2, self.mfi, self.green_dot, self.red_dot = self.I(
            market_cipher_b, self.data.Open, self.data.High, self.data.Low, self.data.Close, self.data.Volume
        )

    def next(self):
        price = self.data.Close[-1]

        # Long entry conditions
        if self.data.Close > self.ema and self.mfi[-1] < self.mfi_oversold and self.green_dot[-1]:
            if not self.position:
                sl = price * (1 - self.sl_pct)
                tp = price * (1 + self.tp_pct)
                self.buy(sl=sl, tp=tp)

        # Short entry conditions
        elif self.data.Close < self.ema and self.mfi[-1] > self.mfi_overbought and self.red_dot[-1]:
            if not self.position:
                sl = price * (1 + self.sl_pct)
                tp = price * (1 - self.tp_pct)
                self.sell(sl=sl, tp=tp)

        # Exit conditions (based on opposite dot)
        if self.position.is_long and self.red_dot[-1]:
            self.position.close()

        if self.position.is_short and self.green_dot[-1]:
            self.position.close()

def sanitize_stats(stats):
    """Sanitizes the stats object for JSON serialization."""
    stats_dict = stats.to_dict()
    # Remove non-serializable objects
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)
    return stats_dict

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    data.columns = [col.strip().capitalize() for col in data.columns]


    # Backtest
    bt = Backtest(data, MarketCipherBImpulseCorrection, cash=100_000, commission=.002)

    stats = bt.run()
    print(stats)

    bt.plot(filename='results/market_cipher_b_impulse_correction.html')

    # Save results
    import json
    sanitized_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)
