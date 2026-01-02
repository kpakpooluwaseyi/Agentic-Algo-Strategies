
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import json
import os
from numpy import isfinite

def custom_indicators(series_dict, macd_fast=12, macd_slow=26, macd_signal=9, mfi_period=14):
    """
    Calculates MACD and MFI using pandas_ta.
    NOTE: This is a proxy for the 'Market Cipher B' or 'VuManchu' indicators
    as the specified custom module 'src.indicators.vumanchu' does not exist.
    """
    close_series = pd.Series(series_dict['Close'])
    high_series = pd.Series(series_dict['High'])
    low_series = pd.Series(series_dict['Low'])
    volume_series = pd.Series(series_dict['Volume'])

    macd_df = ta.macd(close_series, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    macd_line = macd_df[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}']
    macd_signal_line = macd_df[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}']

    mfi = ta.mfi(high=high_series, low=low_series, close=close_series, volume=volume_series, length=mfi_period)

    return macd_line.values, macd_signal_line.values, mfi.values


class StrategyB299cb0a7b10(Strategy):
    """
    A placeholder strategy implementing a MACD crossover system with an MFI filter.
    This serves as a proxy for the requested 'Market Cipher B' strategy due to missing proprietary indicators.
    """
    macd_fast_period = 12
    macd_slow_period = 26
    macd_signal_period = 9

    mfi_period = 14
    mfi_oversold = 20
    mfi_overbought = 80

    stop_loss_pct = 2.0
    take_profit_pct = 5.0

    def init(self):
        self.series_dict = {
            'Close': self.data.Close,
            'High': self.data.High,
            'Low': self.data.Low,
            'Volume': self.data.Volume
        }

        self.macd, self.macd_signal, self.mfi = self.I(
            custom_indicators,
            self.series_dict,
            macd_fast=self.macd_fast_period,
            macd_slow=self.macd_slow_period,
            macd_signal=self.macd_signal_period,
            mfi_period=self.mfi_period,
            name="Custom Indicators"
        )

    def next(self):
        price = self.data.Close[-1]

        is_long_signal = crossover(self.macd, self.macd_signal) and self.mfi[-1] < self.mfi_oversold
        is_short_signal = crossover(self.macd_signal, self.macd) and self.mfi[-1] > self.mfi_overbought

        if self.position.is_long and is_short_signal:
            self.position.close()

        if self.position.is_short and is_long_signal:
            self.position.close()

        if not self.position:
            if is_long_signal:
                sl = price * (1 - self.stop_loss_pct / 100)
                tp = price * (1 + self.take_profit_pct / 100)
                self.buy(sl=sl, tp=tp)
            elif is_short_signal:
                sl = price * (1 + self.stop_loss_pct / 100)
                tp = price * (1 - self.take_profit_pct / 100)
                self.sell(sl=sl, tp=tp)


def sanitize_stats(stats):
    if stats is None:
        return {}
    if isinstance(stats, pd.Series):
        stats = stats.to_dict()
    sanitized = {}
    excluded_keys = ['_equity_curve', '_trades']
    for key, value in stats.items():
        if key in excluded_keys:
            continue
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, float) and not isfinite(value):
            sanitized[key] = None
        elif isinstance(value, (int, float, str, bool, type(None))):
            sanitized[key] = value
        else:
            try:
                json.dumps(value)
                sanitized[key] = value
            except (TypeError, OverflowError):
                sanitized[key] = str(value)
    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data = pd.read_csv(
        data_path,
        index_col=0,
        parse_dates=True,
        header=0,
        names=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
        usecols=[0, 1, 2, 3, 4, 5]
    )

    bt = Backtest(data, StrategyB299cb0a7b10, cash=100000, commission=.002)
    stats = bt.run()
    print(stats)

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    final_results = sanitize_stats(stats)
    final_results['strategy_name'] = 'StrategyB299cb0a7b10'

    results_path = os.path.join(results_dir, 'temp_result.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Backtest stats saved to {results_path}")

    plot_path = os.path.join(results_dir, 'strategy_b299cb0a7b10_plot.html')
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Backtest plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
