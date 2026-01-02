# NOTE: The user's request specified using `src.indicators.vumanchu` for the Market Cipher B indicator.
# However, this module was not found in the codebase. Therefore, this strategy uses a proxy implementation
# built from standard `pandas_ta` indicators (VWAP, MFI, RSI) to approximate the signals.

import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks
from backtesting import Backtest, Strategy

# Helper function to calculate the Market Cipher B proxy indicators
def market_cipher_b_proxy(ohlc: pd.DataFrame, vwap_period=10, trigger_period=5, mfi_period=14, rsi_period=14):
    """
    Calculates proxy indicators for Market Cipher B using standard indicators.
    - VWA Proxy: VWAP
    - Trigger Line Proxy: SMA of VWAP
    - Money Flow Proxy: MFI
    - Momentum Waves Proxy: RSI
    """
    vwap = ta.vwap(high=ohlc['High'], low=ohlc['Low'], close=ohlc['Close'], volume=ohlc['Volume'], length=vwap_period)
    mfi = ta.mfi(high=ohlc['High'], low=ohlc['Low'], close=ohlc['Close'], volume=ohlc['Volume'], length=mfi_period)
    rsi = ta.rsi(close=ohlc['Close'], length=rsi_period)

    # If any indicator is None (e.g., not enough data), create a NaN series to avoid errors
    if vwap is None: vwap = pd.Series(np.nan, index=ohlc.index)
    if mfi is None: mfi = pd.Series(np.nan, index=ohlc.index)
    if rsi is None: rsi = pd.Series(np.nan, index=ohlc.index)

    trigger = ta.sma(vwap, length=trigger_period)
    if trigger is None: trigger = pd.Series(np.nan, index=ohlc.index)

    # For the strategy logic, we'll need to know if the VWAP is trending up or down.
    # We can use a simple diff for this.
    vwap_trend = vwap.diff()

    # Similarly for Money Flow trend
    mfi_trend = mfi.diff()

    return {
        "vwap": vwap,
        "trigger": trigger,
        "mfi": mfi,
        "rsi": rsi,
        "vwap_trend": vwap_trend,
        "mfi_trend": mfi_trend,
    }

class MarketCipherBMultiTimeframeTrend(Strategy):
    """
    A trend-following strategy that uses a proxy for the Market Cipher B indicator
    on a higher timeframe (4h) to determine the macro trend and a lower timeframe (1h)
    for entries on pullbacks.
    """

    # Default parameters for the strategy and indicators
    vwap_period = 10
    trigger_period = 5
    mfi_period = 14
    rsi_period = 14
    risk_reward_ratio = 2.0
    risk_percent = 0.02 # 2% of equity per trade

    def init(self):
        """
        Initialize the strategy, including the multi-timeframe indicators.
        """
        # The data is pre-processed, so we just need to "import" the indicator columns.
        self.htf_vwap = self.I(lambda x: x, self.data.df['4H_vwap'])
        self.htf_trigger = self.I(lambda x: x, self.data.df['4H_trigger'])
        self.htf_vwap_trend = self.I(lambda x: x, self.data.df['4H_vwap_trend'])
        self.htf_mfi_trend = self.I(lambda x: x, self.data.df['4H_mfi_trend'])

        self.etf_vwap = self.I(lambda x: x, self.data.df['1H_vwap'])
        self.etf_trigger = self.I(lambda x: x, self.data.df['1H_trigger'])
        self.etf_mfi = self.I(lambda x: x, self.data.df['1H_mfi'])
        self.etf_rsi = self.I(lambda x: x, self.data.df['1H_rsi'])
        self.etf_mfi_trend = self.I(lambda x: x, self.data.df['1H_mfi_trend'])


    def next(self):
        """
        Define the trading logic for the next bar.
        """
        price = self.data.Close[-1]

        # --- Exit Conditions ---
        if self.position:
            if self.position.is_long:
                # Close long if 1h VWAP crosses below trigger or MFI reverses
                if self.etf_vwap[-1] < self.etf_trigger[-1] or self.etf_mfi_trend[-1] < 0:
                    self.position.close()
            elif self.position.is_short:
                # Close short if 1h VWAP crosses above trigger or MFI reverses
                if self.etf_vwap[-1] > self.etf_trigger[-1] or self.etf_mfi_trend[-1] > 0:
                    self.position.close()

        # --- Entry Conditions (if no position is open) ---
        if not self.position:
            # Long Entry Conditions
            htf_long_cond = (self.htf_vwap[-1] > self.htf_trigger[-1] and
                             self.htf_vwap_trend[-1] > 0 and
                             self.htf_mfi_trend[-1] > 0)

            etf_long_cond = (self.etf_vwap[-1] > self.etf_trigger[-1] and
                             self.etf_mfi[-1] > 50 and
                             self.etf_rsi[-1] > 50 and
                             self.etf_mfi_trend[-1] > 0)

            # Pullback condition: price touches the 1h VWAP
            pullback_long_cond = self.data.Low[-1] <= self.etf_vwap[-1]

            if htf_long_cond and etf_long_cond and pullback_long_cond:
                swing_low = get_last_swing(self.data.Low, is_high=False)
                if swing_low is not None and swing_low < price:
                    sl = swing_low
                    tp = price + (price - sl) * self.risk_reward_ratio
                    size = (self.equity * self.risk_percent) / (price - sl)
                    if size > 0:
                        self.buy(sl=sl, tp=tp, size=int(size))

            # Short Entry Conditions
            htf_short_cond = (self.htf_vwap[-1] < self.htf_trigger[-1] and
                              self.htf_vwap_trend[-1] < 0 and
                              self.htf_mfi_trend[-1] < 0)

            etf_short_cond = (self.etf_vwap[-1] < self.etf_trigger[-1] and
                              self.etf_mfi[-1] < 50 and
                              self.etf_rsi[-1] < 50 and
                              self.etf_mfi_trend[-1] < 0)

            # Bounce condition: price touches the 1h VWAP
            bounce_short_cond = self.data.High[-1] >= self.etf_vwap[-1]

            if htf_short_cond and etf_short_cond and bounce_short_cond:
                swing_high = get_last_swing(self.data.High, is_high=True)
                if swing_high is not None and swing_high > price:
                    sl = swing_high
                    tp = price - (sl - price) * self.risk_reward_ratio
                    size = (self.equity * self.risk_percent) / (sl - price)
                    if size > 0:
                        self.sell(sl=sl, tp=tp, size=int(size))

def get_last_swing(series, is_high=True, lookback=50):
    """
    Finds the most recent swing high or low.
    """
    series = series[-lookback:]
    if is_high:
        peaks, _ = find_peaks(series)
        if len(peaks) > 0:
            return series[peaks[-1]]
    else:
        peaks, _ = find_peaks(-series)
        if len(peaks) > 0:
            return series[peaks[-1]]
    return None

def add_multi_timeframe_indicators(df, params):
    """
    Pre-calculates indicators on 1H and 4H timeframes and merges them back into the main DataFrame.
    """
    ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}

    # --- 4H Indicators ---
    df_4h = df.resample('4H').agg(ohlc_dict).dropna()
    proxy_4h = market_cipher_b_proxy(df_4h, params.vwap_period, params.trigger_period, params.mfi_period, params.rsi_period)
    for key, value in proxy_4h.items():
        df[f'4H_{key}'] = value.reindex(df.index, method='ffill')

    # --- 1H Indicators ---
    df_1h = df.resample('1H').agg(ohlc_dict).dropna()
    proxy_1h = market_cipher_b_proxy(df_1h, params.vwap_period, params.trigger_period, params.mfi_period, params.rsi_period)
    for key, value in proxy_1h.items():
        df[f'1H_{key}'] = value.reindex(df.index, method='ffill')

    return df

# --- Backtesting setup ---
if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)

    # Ensure columns are named correctly for backtesting.py
    data.columns = [col.strip().capitalize() for col in data.columns]

    # Pre-calculate indicators
    data = add_multi_timeframe_indicators(data, MarketCipherBMultiTimeframeTrend)

    # Instantiate and run the backtest
    bt = Backtest(data, MarketCipherBMultiTimeframeTrend, cash=100_000, commission=.002)
    stats = bt.run()

    print(stats)

    # Save stats to a JSON file
    import json

    # Sanitize stats for JSON serialization
    def sanitize_stats(stats):
        # Remove non-serializable objects
        if '_strategy' in stats:
            del stats['_strategy']
        if '_equity_curve' in stats:
            del stats['_equity_curve']
        if '_trades' in stats:
            del stats['_trades']

        # Convert any remaining non-serializable types
        for key, value in stats.items():
            if isinstance(value, pd.Timestamp):
                stats[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                stats[key] = str(value)
            elif pd.isna(value):
                stats[key] = None
        return stats

    sanitized_stats = sanitize_stats(stats.to_dict())

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    # Generate the plot
    try:
        bt.plot(filename='results/market_cipher_b_multi_timeframe_trend.html')
    except Exception as e:
        print(f"Could not generate plot: {e}")
