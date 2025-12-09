import pandas as pd
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

class FibonacciCenterPeakScalpStrategy(Strategy):
    """
    This strategy identifies and trades M-patterns using a pre-processing step
    within the init method, making it compatible with backtesting.py's optimization.
    """
    # Optimizable parameters
    peak_prominence = 1.0
    fib_level_min = 0.45
    fib_level_max = 0.55

    def init(self):
        """
        Pre-process the data to find M-patterns and store trade signals.
        This method is called once per optimization run.
        """
        highs = pd.Series(self.data.High)
        lows = pd.Series(self.data.Low)

        peak_indices, _ = find_peaks(highs, prominence=self.peak_prominence)
        trough_indices, _ = find_peaks(-lows, prominence=self.peak_prominence)

        entry_signals = np.zeros(len(self.data))
        stop_losses = np.full(len(self.data), np.nan)
        take_profits = np.full(len(self.data), np.nan)

        for i in range(len(peak_indices) - 1):
            for j in range(len(trough_indices)):
                peak_a_idx = int(peak_indices[i])
                trough_b_idx = int(trough_indices[j])
                peak_c_idx = int(peak_indices[i+1])

                if not (peak_a_idx < trough_b_idx < peak_c_idx):
                    continue

                peak_a_price = highs.iloc[peak_a_idx]
                trough_b_price = lows.iloc[trough_b_idx]
                peak_c_price = highs.iloc[peak_c_idx]

                if peak_c_price >= peak_a_price:
                    continue

                ab_range = peak_a_price - trough_b_price
                bc_retracement = peak_c_price - trough_b_price
                fib_level = bc_retracement / ab_range if ab_range > 0 else 0

                if not (self.fib_level_min <= fib_level <= self.fib_level_max):
                    continue

                signal_idx = peak_c_idx + 1
                if signal_idx < len(self.data):
                    entry_signals[signal_idx] = 1
                    stop_losses[signal_idx] = peak_c_price * 1.001
                    take_profits[signal_idx] = peak_c_price - (peak_c_price - trough_b_price) * 0.5

                break

        self.entry_signal = self.I(lambda x: x, entry_signals)
        self.sl = self.I(lambda x: x, stop_losses)
        self.tp = self.I(lambda x: x, take_profits)

    def next(self):
        if self.entry_signal[-1] == 1 and not self.position:
            sl = self.sl[-1]
            tp = self.tp[-1]
            if not np.isnan(sl) and not np.isnan(tp) and tp < self.data.Close[-1]:
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    n = 200
    index = pd.date_range('2023-01-01', periods=n, freq='h')

    peak_a_idx, peak_a_price = 50, 110
    trough_b_idx, trough_b_price = 100, 100
    peak_c_idx, peak_c_price = 150, 105

    price = np.zeros(n)
    price[0:peak_a_idx] = np.linspace(105, peak_a_price, peak_a_idx)
    price[peak_a_idx-1:trough_b_idx] = np.linspace(peak_a_price, trough_b_price, trough_b_idx - (peak_a_idx-1))
    price[trough_b_idx-1:peak_c_idx] = np.linspace(trough_b_price, peak_c_price, peak_c_idx - (trough_b_idx-1))
    price[peak_c_idx-1:n] = np.linspace(peak_c_price, peak_c_price - 10, n - (peak_c_idx-1))

    np.random.seed(42)
    noise = np.random.uniform(-0.1, 0.1, n)
    price += noise

    data = pd.DataFrame(index=index)
    data['Open'] = price
    data['Close'] = price
    data['High'] = price + 0.1
    data['Low'] = price - 0.1

    bt = Backtest(data, FibonacciCenterPeakScalpStrategy, cash=100_000, commission=.002)

    stats = bt.optimize(
        peak_prominence=list(np.arange(0.1, 1.0, 0.1)),
        fib_level_min=list(np.arange(0.4, 0.5, 0.02)),
        fib_level_max=list(np.arange(0.5, 0.6, 0.02)),
        maximize='Sharpe Ratio',
        return_heatmap=False
    )

    print("--- Optimization Stats ---")
    print(stats)
    print("\n--- Best Parameters ---")
    print(stats._strategy)
    print("\n--- Trades ---")
    print(stats['_trades'])

    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'fibonacci_center_peak_scalp',
            'return': float(stats.get('Return [%]', 0.0)),
            'sharpe': float(stats.get('Sharpe Ratio', 0.0) if np.isnan(stats.get('Sharpe Ratio', 0.0)) else stats.get('Sharpe Ratio', 0.0)),
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
            'win_rate': float(stats.get('Win Rate [%]', 0.0)),
            'total_trades': int(stats.get('# Trades', 0))
        }, f, indent=2)

    print("\nResults saved to results/temp_result.json")
    bt.plot(filename="results/fibonacci_strategy_plot.html", open_browser=False)
