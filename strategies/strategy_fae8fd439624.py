
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy
from backtesting.lib import crossover, FractionalBacktest

# Custom indicator function for WaveTrend-like oscillator
def wt(high, low, close, channel_length=9, average_length=12):
    """
    WaveTrend Oscillator Proxy
    Combines EMAs of the average and range of the price to detect trends.
    Fills initial NaN values to prevent errors.
    """
    ap = (high + low + close) / 3
    esa = ta.ema(ap, length=channel_length)

    # Fill initial NaNs in esa before subtraction
    esa = esa.bfill()

    d = ta.ema(abs(ap - esa), length=channel_length)
    # Fill initial NaNs in d
    d = d.bfill()

    # Prevent division by zero if d is 0
    d.replace(0, 1e-9, inplace=True)

    ci = (ap - esa) / (0.015 * d)
    tci = ta.ema(ci, length=average_length)

    # Fill any remaining NaNs at the start of the final series
    tci = tci.bfill()
    return tci.values # Return numpy array as expected by self.I

class StrategyFae8fd439624(Strategy):
    """
    Proxy strategy for 'Market Cipher B' / 'VuManchu' indicator.
    The requested 'src.indicators.vumanchu' was not found. This strategy
    approximates its behavior using a combination of a WaveTrend-like
    oscillator and the Money Flow Index (MFI) from pandas_ta.

    Entry Conditions:
    - Long: WaveTrend crosses above a scaled MFI when both are below a lower threshold.
    - Short: WaveTrend crosses below a scaled MFI when both are above an upper threshold.
    """
    wt_channel_length = 10
    wt_average_length = 21
    mfi_period = 14
    upper_threshold = 40
    lower_threshold = -50
    sl_pct = 0.05  # Stop loss percentage
    tp_pct = 0.10  # Take profit percentage

    def init(self):
        # Convert data to pandas Series for pandas_ta
        high_s = pd.Series(self.data.High)
        low_s = pd.Series(self.data.Low)
        close_s = pd.Series(self.data.Close)
        volume_s = pd.Series(self.data.Volume)

        # Calculate the WaveTrend-like oscillator
        self.wt_oscillator = self.I(wt, high_s, low_s, close_s,
                                    channel_length=self.wt_channel_length,
                                    average_length=self.wt_average_length)

        # Calculate the Money Flow Index
        mfi_series = ta.mfi(high_s, low_s, close_s, volume_s, length=self.mfi_period).bfill()

        # Scale MFI from 0-100 to a range more comparable to WaveTrend (-100 to 100)
        self.mfi_scaled = self.I(lambda x: (x - 50) * 2, mfi_series.values)


    def next(self):
        wt_val = self.wt_oscillator[-1]
        mfi_val = self.mfi_scaled[-1]
        price = self.data.Close[-1]

        # Long entry conditions
        if (crossover(self.wt_oscillator, self.mfi_scaled) and
            wt_val < self.lower_threshold and
            mfi_val < self.lower_threshold):
            if not self.position:
                sl = price * (1 - self.sl_pct)
                tp = price * (1 + self.tp_pct)
                self.buy(size=0.1, sl=sl, tp=tp) # Trade 10% of equity

        # Short entry conditions
        elif (crossover(self.mfi_scaled, self.wt_oscillator) and
              wt_val > self.upper_threshold and
              mfi_val > self.upper_threshold):
            if not self.position:
                sl = price * (1 + self.sl_pct)
                tp = price * (1 - self.tp_pct)
                self.sell(size=0.1, sl=sl, tp=tp) # Trade 10% of equity

if __name__ == '__main__':
    import os
    import json
    import numpy as np

    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.strip().title() for c in data.columns]

    # Use FractionalBacktest for crypto trading
    bt = FractionalBacktest(data, StrategyFae8fd439624, cash=10000, commission=.002, finalize_trades=True)

    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        """A helper function to sanitize backtesting stats for JSON serialization."""
        sanitized = {}
        for key, value in stats.items():
            if key in ('_strategy', '_equity_curve', '_trades'):
                continue

            # Check for pandas NA, which is more general
            if pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (np.integer, np.int64)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sanitized[key] = float(value)
            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                 sanitized[key] = str(value)
            else:
                sanitized[key] = value
        return sanitized

    clean_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(clean_stats, f, indent=4)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    try:
        plot_filename = 'results/strategy_fae8fd439624.html'
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
