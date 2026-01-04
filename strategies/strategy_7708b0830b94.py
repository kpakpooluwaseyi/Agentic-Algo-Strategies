
import pandas as pd
import numpy as np
import talib
from scipy.signal import find_peaks
from backtesting import Strategy, Backtest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indicators.vumanchu import cipher_b


def find_divergence(price, indicator, order=5, min_dist=5):
    """
    Finds bullish and bearish divergences between price and an indicator.

    Returns two boolean Series: bullish_div, bearish_div
    """
    # Find peaks and troughs
    price_peaks, _ = find_peaks(price, distance=min_dist, prominence=(price.std() / 10))
    price_troughs, _ = find_peaks(-price, distance=min_dist, prominence=(price.std() / 10))
    indicator_peaks, _ = find_peaks(indicator, distance=min_dist, prominence=(indicator.std() / 10))
    indicator_troughs, _ = find_peaks(-indicator, distance=min_dist, prominence=(indicator.std() / 10))

    bullish_div = pd.Series(False, index=price.index)
    bearish_div = pd.Series(False, index=price.index)

    # Bearish Divergence: Higher high in price, lower high in indicator
    for i in range(1, len(price_peaks)):
        for j in range(1, len(indicator_peaks)):
            if abs(price_peaks[i] - indicator_peaks[j]) < 5 and abs(price_peaks[i-1] - indicator_peaks[j-1]) < 5:
                if price[price_peaks[i]] > price[price_peaks[i-1]] and \
                   indicator[indicator_peaks[j]] < indicator[indicator_peaks[j-1]]:
                    bearish_div.iloc[price_peaks[i]] = True

    # Bullish Divergence: Lower low in price, higher low in indicator
    for i in range(1, len(price_troughs)):
        for j in range(1, len(indicator_troughs)):
             if abs(price_troughs[i] - indicator_troughs[j]) < 5 and abs(price_troughs[i-1] - indicator_troughs[j-1]) < 5:
                if price[price_troughs[i]] < price[price_troughs[i-1]] and \
                   indicator[indicator_troughs[j]] > indicator[indicator_troughs[j-1]]:
                    bullish_div.iloc[price_troughs[i]] = True

    return bullish_div, bearish_div

def preprocess_data(df, **params):
    """
    Adds all indicators and signals to the dataframe.
    """
    df = df.copy()

    # Apply Cipher B indicators
    df = cipher_b(df)

    # Clean column names
    df.columns = [c.strip().title() for c in df.columns]

    # Mandatory features from guidelines
    # 1. ATR
    df['Atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # 2. Higher Timeframe Trend Filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['Ema200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df['Htf_Uptrend'] = (df_4h['Close'] > df_4h['Ema200']).reindex(df.index, method='ffill').fillna(False)

    # 3. Volume Confirmation
    df['Volume_Sma'] = talib.SMA(df['Volume'], timeperiod=20)

    # Divergence calculation
    # Ensure no NaNs are passed to the divergence function
    df.dropna(inplace=True)

    bullish_div_wt, bearish_div_wt = find_divergence(df['Close'], df['Wt1'])
    bullish_div_mfi, bearish_div_mfi = find_divergence(df['Close'], df['Rsimfi'])

    df['Bullish_Divergence'] = bullish_div_wt & bullish_div_mfi
    df['Bearish_Divergence'] = bearish_div_wt & bearish_div_mfi

    return df


class MarketCipherScalp(Strategy):
    """
    Scalping strategy based on Market Cipher B divergences, with mandatory filters.
    """
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        # Indicators
        self.atr = self.I(lambda: self.data.Atr, name='ATR')
        self.htf_uptrend = self.I(lambda: self.data.Htf_Uptrend, name='HTF_Uptrend')
        self.volume_sma = self.I(lambda: self.data.Volume_Sma, name='Volume_SMA')
        self.bullish_divergence = self.I(lambda: self.data.Bullish_Divergence, name='Bullish_Div')
        self.bearish_divergence = self.I(lambda: self.data.Bearish_Divergence, name='Bearish_Div')

    def next(self):
        price = self.data.Close[-1]

        # --- FILTERS ---
        # Volume filter
        if self.data.Volume[-1] < self.volume_sma[-1]:
            return

        # --- ENTRY LOGIC ---
        if not self.position:
            # Bullish Entry: Bullish divergence in an uptrend
            if self.bullish_divergence[-1] and self.htf_uptrend[-1]:
                sl = price - self.atr[-1] * self.atr_sl_multiplier
                tp = price + self.atr[-1] * self.atr_tp_multiplier
                self.buy(sl=sl, tp=tp)

            # Bearish Entry: Bearish divergence in a downtrend
            elif self.bearish_divergence[-1] and not self.htf_uptrend[-1]:
                sl = price + self.atr[-1] * self.atr_sl_multiplier
                tp = price - self.atr[-1] * self.atr_tp_multiplier
                self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    # Load data
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # As a fallback, try to generate some synthetic data for testing
        print("Generating synthetic data for testing purposes...")
        from backtesting.test import EURUSD
        df = EURUSD.copy()
        df = df.iloc[-5000:]
        df.columns = [c.title() for c in df.columns]

    # Preprocess the data
    df = preprocess_data(df)

    # Initialize and run the backtest
    bt = Backtest(df, MarketCipherScalp, cash=100_000, commission=.002)
    stats = bt.run()

    # Print the results
    print(stats)

    # Save results to a JSON file
    stats_dict = dict(stats)

    # Sanitize the stats dict for JSON serialization
    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            stats_dict[key] = float(value)
        elif isinstance(value, pd.DataFrame):
            stats_dict[key] = value.to_dict()
        elif isinstance(value, type(pd.NA)):
            stats_dict[key] = None

    # Remove non-serializable objects
    if '_strategy' in stats_dict:
        del stats_dict['_strategy']
    if '_equity_curve' in stats_dict:
        del stats_dict['_equity_curve']
    if '_trades' in stats_dict:
        del stats_dict['_trades']

    output_path = 'results/temp_result.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        import json
        json.dump(stats_dict, f, indent=4)

    print(f"Results saved to {output_path}")

    # Generate the plot
    plot_filename = 'results/strategy_7708b0830b94.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")
