import pandas as pd
from backtesting import Backtest, Strategy
import talib
import os
import json
import numpy as np

class SmbSectorBreakoutLong(Strategy):
    """
    A breakout strategy focusing on strength in leading sectors.
    Goes long on a breakout over a short-term resistance level.
    """
    # --- Strategy Parameters ---
    ma_short_period = 20
    ma_medium_period = 50
    resistance_lookback = 20
    ma_convergence_threshold = 0.05 # Increased from 0.01 to 0.05
    risk_reward_ratio = 2.0 # Added for take-profit

    def init(self):
        """
        Initialize the strategy.
        """
        # --- Indicators ---
        self.ma_short = self.I(talib.SMA, self.data.Close, self.ma_short_period)
        self.ma_medium = self.I(talib.SMA, self.data.Close, self.ma_medium_period)
        self.resistance = self.I(lambda x: pd.Series(x).rolling(self.resistance_lookback).max(), self.data.High)

    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        price = self.data.Close[-1]

        ma_spread = abs(self.ma_short[-1] - self.ma_medium[-1]) / price
        are_mas_converged = ma_spread < self.ma_convergence_threshold

        is_bullish_context = (
            price > self.ma_short[-1] and
            price > self.ma_medium[-1] and
            self.ma_short[-1] > self.ma_medium[-1] and
            are_mas_converged
        )

        is_breakout = price > self.resistance[-2]

        if not self.position and is_bullish_context and is_breakout:
            stop_loss = self.data.Low[-1]
            take_profit = price + (price - stop_loss) * self.risk_reward_ratio
            self.buy(sl=stop_loss, tp=take_profit)


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        print("Data file not found. Generating synthetic data...")
        n_points = 5000
        index = pd.to_datetime(pd.date_range('2022-01-01', periods=n_points, freq='15min'))
        price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.1)
        price[2000:3000] = 105
        price[3000:] += 10
        data = pd.DataFrame({
            'Open': price, 'High': price + 0.5, 'Low': price - 0.5, 'Close': price, 'Volume': np.random.uniform(100, 500, n_points)
        }, index=index)
    else:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [col.strip().capitalize() for col in data.columns]

    bt = Backtest(data, SmbSectorBreakoutLong, cash=100_000, commission=.002, finalize_trades=True)

    print("Running backtest with refined parameters...")
    stats = bt.run()

    print("Backtest Stats:")
    print(stats)

    os.makedirs('results', exist_ok=True)

    results = {
        'strategy_name': 'smb_sector_breakout_long',
        'return': stats.get('Return [%]', None),
        'sharpe': stats.get('Sharpe Ratio', None),
        'max_drawdown': stats.get('Max. Drawdown [%]', None),
        'win_rate': stats.get('Win Rate [%]', None),
        'total_trades': stats.get('# Trades', 0)
    }

    cleaned_results = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in results.items()}

    with open('results/temp_result.json', 'w') as f:
        json.dump(cleaned_results, f, indent=2)
        f.write('\n')

    print("Results saved to results/temp_result.json")

    try:
        plot_filename = 'results/smb_sector_breakout_long.html'
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
