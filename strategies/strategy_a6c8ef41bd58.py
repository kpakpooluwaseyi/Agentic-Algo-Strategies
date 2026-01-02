# Proxy implementation for the Vumanchu Cipher B strategy
# This strategy uses MFI and RSI as proxies for the proprietary indicators,
# as the original 'src.indicators.vumanchu' was not found in the repository.

from backtesting import Backtest, Strategy
import pandas as pd
import pandas_ta as ta

# Wrapper functions for pandas-ta indicators to make them compatible with backtesting.py
def mfi(high, low, close, volume, length):
    series = ta.mfi(
        high=pd.Series(high),
        low=pd.Series(low),
        close=pd.Series(close),
        volume=pd.Series(volume),
        length=length
    )
    return series.values

def rsi(close, length):
    series = ta.rsi(
        close=pd.Series(close),
        length=length
    )
    return series.values

def sma(close, length):
    series = ta.sma(
        close=pd.Series(close),
        length=length
    )
    return series.values


class VumanchuCipherBProxy(Strategy):
    """
    Proxy implementation for a VuManchu / Market Cipher B style strategy.
    It uses a combination of Money Flow Index (MFI) for volume flow,
    Relative Strength Index (RSI) for momentum, and a Simple Moving
    Average (SMA) as a trend filter.
    """
    # Indicator parameters
    mfi_period = 14
    rsi_period = 14
    sma_period = 50
    oversold_mfi = 20
    oversold_rsi = 30
    overbought_mfi = 80
    overbought_rsi = 70

    # Risk management parameters
    sl_pct = 0.02  # 2% stop loss
    tp_pct = 0.04  # 4% take profit

    def init(self):
        # Calculate indicators using the wrapper functions
        self.mfi = self.I(mfi, self.data.High, self.data.Low, self.data.Close, self.data.Volume, self.mfi_period)
        self.rsi = self.I(rsi, self.data.Close, self.rsi_period)
        self.sma = self.I(sma, self.data.Close, self.sma_period)

    def next(self):
        price = self.data.Close[-1]

        # Trend filter
        is_uptrend = price > self.sma[-1]
        is_downtrend = price < self.sma[-1]

        # Entry conditions
        long_condition = (
            (self.mfi[-1] < self.oversold_mfi or self.rsi[-1] < self.oversold_rsi) and
            is_uptrend
        )

        short_condition = (
            (self.mfi[-1] > self.overbought_mfi or self.rsi[-1] > self.overbought_rsi) and
            is_downtrend
        )

        # Execute trades
        if not self.position:
            if long_condition:
                sl = price * (1 - self.sl_pct)
                tp = price * (1 + self.tp_pct)
                self.buy(sl=sl, tp=tp)
            elif short_condition:
                sl = price * (1 + self.sl_pct)
                tp = price * (1 - self.tp_pct)
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    import os
    import json

    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Clean up column names: strip whitespace, convert to lowercase, then capitalize
    data.columns = [c.strip().lower() for c in data.columns]
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    # Ensure datetime index is correct and remove any columns that are not OHLCV
    ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[[col for col in ohlcv_columns if col in data.columns]]

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    bt = Backtest(data, VumanchuCipherBProxy, cash=100000, commission=.002)

    print("Running backtest...")
    stats = bt.run()
    print(stats)

    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Sanitize stats for JSON serialization
    def sanitize_stats(stats_series):
        sanitized = {}
        for key, value in stats_series.items():
            if isinstance(key, str) and key.startswith('_'):
                continue
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (int, float)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    sanitized_stats = sanitize_stats(stats)

    result_path = os.path.join(results_dir, 'temp_result.json')
    with open(result_path, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print(f"Results saved to {result_path}")

    # Generate plot
    plot_path = os.path.join(results_dir, 'strategy_a6c8ef41bd58.html')
    bt.plot(filename=plot_path, open_browser=False)
    print(f"Plot saved to {plot_path}")
