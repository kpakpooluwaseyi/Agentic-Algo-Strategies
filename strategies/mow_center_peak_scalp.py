from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import json

def EMA(series, period):
    """Custom EMA indicator function for backtesting.py"""
    return pd.Series(series).ewm(span=period, adjust=False).mean().values

class MowCenterPeakScalpStrategy(Strategy):
    ema_period = 50

    def init(self):
        self.ema = self.I(EMA, self.data.Close, self.ema_period)

    def next(self):
        if not self.position:
            if self.data.Close[-1] > self.ema[-1]:
                self.buy()
        elif self.data.Close[-1] < self.ema[-1]:
            self.position.close()

if __name__ == '__main__':
    from backtesting.test import GOOG
    data = GOOG.tail(2000) # Use a smaller subset for faster testing

    bt = Backtest(data, MowCenterPeakScalpStrategy, cash=100_000, commission=.002)

    stats = bt.optimize(ema_period=range(20, 100, 10), maximize='Sharpe Ratio')

    import os
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        stats_dict = stats.to_dict()
        results_dict = {
            'strategy_name': 'mow_center_peak_scalp',
            'return': stats_dict.get('Return [%]', 0.0),
            'sharpe': stats_dict.get('Sharpe Ratio', 0.0),
            'max_drawdown': stats_dict.get('Max. Drawdown [%]', 0.0),
            'win_rate': stats_dict.get('Win Rate [%]', 0.0),
            'total_trades': int(stats_dict.get('# Trades', 0))
        }
        clean_stats = {k: (v if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else 0.0) for k, v in results_dict.items()}
        json.dump(clean_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    try:
        bt.plot(filename="results/mow_center_peak_scalp.html")
        print("Plot saved to results/mow_center_peak_scalp.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
