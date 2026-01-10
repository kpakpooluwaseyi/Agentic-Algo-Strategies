"""
Strategy Template for Backtesting Pipeline
==========================================

Instructions:
1. Rename this file to your_strategy_name.py (use underscores, lowercase)
2. Rename the class to YourStrategyName (CamelCase)
3. Implement your trading logic in next()
4. Drop the file in strategies/ folder
5. The local_runner will automatically pick it up and test it

The pipeline will:
- Run your strategy on 6 BTC timeframes (4h, 1h, 15m, 5m, 1m)
- Perform Walk-Forward Analysis (WFA) with 30% out-of-sample data
- Perform Walk-Forward Optimization (WFO) if WFA fails
- Add results to the leaderboard
"""

import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import json
import os

def sanitize_stats(stats):
    """Sanitizes the backtest stats object for JSON serialization."""
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (int, float, bool, str, type(None))):
            sanitized[key] = value
        elif key in ['_strategy', '_equity_curve', '_trades']:
            continue
    return sanitized

def preprocess_data(df):
    """Adds Bollinger Bands and ATR indicators to the DataFrame."""
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.dropna(inplace=True)
    return df

class BollingerBandMeanReversion(Strategy):
    """
    A mean-reversion strategy that enters trades when the price touches
    the outer Bollinger Bands and uses ATR for risk management, as required
    by MoonDev strategy guidelines.

    Entry Conditions:
    - Long: Price closes below the lower Bollinger Band.
    - Short: Price closes above the upper Bollinger Band.

    Exit Conditions:
    - Stop Loss: 2 * ATR from the entry price.
    - Take Profit: 3 * ATR from the entry price.
    """
    bbands_length = 20
    bbands_std = 2.0
    atr_period = 14
    sl_multiplier = 2.0
    tp_multiplier = 3.0

    def init(self):
        # Indicators from pandas_ta are automatically available on self.data
        # We just need to access them. self.I is used for plotting.
        self.lower_band = self.I(lambda: self.data.df[f'BBL_{self.bbands_length}_{self.bbands_std}'])
        self.upper_band = self.I(lambda: self.data.df[f'BBU_{self.bbands_length}_{self.bbands_std}'])
        self.atr = self.I(lambda: self.data.df[f'ATRr_{self.atr_period}'])

    def next(self):
        price = self.data.Close[-1]
        atr_value = self.atr[-1]

        # Long entry condition
        if not self.position and price < self.lower_band[-1]:
            sl = price - self.sl_multiplier * atr_value
            tp = price + self.tp_multiplier * atr_value
            self.buy(sl=sl, tp=tp)

        # Short entry condition
        elif not self.position and price > self.upper_band[-1]:
            sl = price + self.sl_multiplier * atr_value
            tp = price - self.tp_multiplier * atr_value
            self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Use a smaller slice of data for faster testing
        df = df.iloc[-5000:]

        df = preprocess_data(df)

        bt = Backtest(df, BollingerBandMeanReversion, cash=100000, commission=.002, finalize_trades=True)

        print("Running backtest...")
        stats = bt.run()
        print(stats)

        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        sanitized = sanitize_stats(stats)
        with open(os.path.join(results_dir, 'temp_result.json'), 'w') as f:
            json.dump(sanitized, f, indent=4)
        print(f"\nResults saved to {results_dir}/temp_result.json")

        plot_filename = os.path.join(results_dir, 'bollinger_band_mean_reversion.html')
        try:
            bt.plot(filename=plot_filename)
            print(f"Plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot due to error: {e}")
