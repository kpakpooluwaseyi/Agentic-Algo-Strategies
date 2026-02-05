
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks
import os
import sys

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame, **params):
    """
    Adds all indicators and required features to the dataframe.
    """
    # Sanitize column names
    df.columns = [col.strip().capitalize() for col in df.columns]

    # -- Parameters --
    peak_distance = params.get('peak_distance', 50)

    # -- Mandatory Guideline Features --
    # 1. Higher Timeframe Trend Filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['ema_200'] = ta.ema(df_4h['Close'], length=200)
    df_4h['htf_uptrend'] = (df_4h['Close'] > df_4h['ema_200']).astype(bool)
    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill').fillna(False)

    # 2. ATR for Risk Management
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # 3. Volume Confirmation
    df['volume_ma'] = ta.sma(df['Volume'], length=20)

    # -- Strategy-Specific Indicators --
    # 1. MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']

    # 2. Stochastic RSI (from vumanchu)
    df = cipher_b(df) # Adds stoch_rsi_k, stoch_rsi_d

    # 3. Support & Resistance Zones
    high_peaks_indices, _ = find_peaks(df['High'], distance=peak_distance)
    low_troughs_indices, _ = find_peaks(-df['Low'], distance=peak_distance)

    df['resistance'] = pd.Series(np.nan, index=df.index)
    df['support'] = pd.Series(np.nan, index=df.index)

    df.iloc[high_peaks_indices, df.columns.get_loc('resistance')] = df.iloc[high_peaks_indices]['High']
    df.iloc[low_troughs_indices, df.columns.get_loc('support')] = df.iloc[low_troughs_indices]['Low']

    df['resistance'] = df['resistance'].ffill()
    df['support'] = df['support'].ffill()

    # The backtesting library handles initial NaN values, so a blanket dropna()
    # can unnecessarily empty the dataframe if the warmup period is long.
    # df.dropna(inplace=True)
    return df

class MacdStochRsiSrReversal(Strategy):
    """
    Strategy based on MACD, Stochastic RSI, and Support/Resistance zones.
    """
    # NOTE: Implementation will be done in a future step.
    def init(self):
        pass

    def next(self):
        pass

if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Data file not found. A sample dataset will be generated.")
        # Generate synthetic data for testing if the file is not available
        dates = pd.date_range(start='2023-01-01', periods=5000, freq='15min')
        np.random.seed(42)
        price = 20000 + np.cumsum(np.random.randn(5000) * 2)
        df = pd.DataFrame({
            'open': price,
            'high': price + np.random.uniform(0, 10, 5000),
            'low': price - np.random.uniform(0, 10, 5000),
            'close': price + np.random.randn(5000),
            'volume': np.random.uniform(10, 500, 5000)
        }, index=dates)
        df.index.name = 'datetime'

    # Preprocess data
    df_processed = preprocess_data(df.copy())

    # Run backtest
    # NOTE: The strategy class is just a placeholder for now.
    bt = Backtest(df_processed, MacdStochRsiSrReversal, cash=100_000, commission=.002)
    stats = bt.run()

    print("Backtest Stats:")
    print(stats)

    # Save results and plot
    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save stats to a JSON file
    stats_df = pd.DataFrame([stats]).drop(columns=['_trades', '_equity_curve', '_strategy'], errors='ignore')
    stats_df.to_json("results/temp_result.json", orient="records", lines=True)

    # Generate plot
    plot_filename = 'results/strategy_038ed62e959d.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")
