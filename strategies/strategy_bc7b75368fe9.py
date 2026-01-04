import pandas as pd
import pandas_ta as ta
from backtesting import Strategy
from backtesting import Backtest
import json
import os
import numpy as np

class StrategyBC7B75368FE9(Strategy):
    # Default parameters
    fast_period = 12
    slow_period = 26
    signal_period = 9
    mfi_period = 14
    vwap_period = 20
    rr_ratio = 2.0
    sl_atr_multiplier = 1.0
    atr_period = 14

    def init(self):
        # Indicators are pre-calculated and available on the data object.
        pass

    def next(self):
        price = self.data.Close[-1]

        # Indicator values from pre-calculated columns
        # Column names are based on default parameters.
        macd_hist = self.data['MACDh_12_26_9'][-1]
        mfi = self.data['MFI_14'][-1]
        vwap = self.data['VWAP_D'][-1]
        atr = self.data['ATRr_14'][-1]

        # Entry conditions
        long_condition = (macd_hist > 0 and mfi < 30 and price > vwap)
        short_condition = (macd_hist < 0 and mfi > 70 and price < vwap)

        if not self.position:
            if long_condition:
                sl = price - atr * self.sl_atr_multiplier
                tp = price + (price - sl) * self.rr_ratio
                self.buy(sl=sl, tp=tp)
            elif short_condition:
                sl = price + atr * self.sl_atr_multiplier
                tp = price - (sl - price) * self.rr_ratio
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    # Load data
    data_path = 'data/BTC-USD-15m.csv'
    data = pd.read_csv(
        data_path,
        index_col=0,
        parse_dates=True,
        header=0,
        names=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
        usecols=[0, 1, 2, 3, 4, 5]
    )
    data.sort_index(inplace=True)

    # Calculate indicators using default parameters from the class
    s = StrategyBC7B75368FE9
    data.ta.macd(fast=s.fast_period, slow=s.slow_period, signal=s.signal_period, append=True)
    data.ta.mfi(length=s.mfi_period, append=True)
    data.ta.vwap(length=s.vwap_period, append=True)
    data.ta.atr(length=s.atr_period, append=True)

    # Drop rows with NaN values from indicator lookback
    data.dropna(inplace=True)

    # Run backtest
    bt = Backtest(data, StrategyBC7B75368FE9, cash=100_000, commission=.002)
    stats = bt.run()

    # Save results
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        # Convert stats Series to a dictionary
        stats_dict = stats.to_dict()

        # Remove non-serializable objects
        stats_dict.pop('_trades', None)
        stats_dict.pop('_equity_curve', None)
        stats_dict.pop('_strategy', None)

        # Convert any remaining non-JSON-serializable types
        for key, value in stats_dict.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                stats_dict[key] = str(value)
            elif pd.isna(value):
                stats_dict[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                stats_dict[key] = float(value)

        return stats_dict

    sanitized_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=2)

    print(stats)
    bt.plot(filename='results/strategy_bc7b75368fe9.html')
