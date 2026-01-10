"""
Cross-Sectional Fundamental Factors (Proxy) Strategy
=====================================================
This strategy is a proxy for the requested cross-sectional fundamental factor model.
Since fundamental data for a single instrument (BTC-USD) is not applicable, this
implementation uses technical indicators as "factors" to predict future returns.

Proxy Logic:
1.  A set of technical indicators (RSI, MACD, Bollinger Bands, etc.) are generated
    to serve as predictive features.
2.  A linear regression model is trained on the first 70% of the data to learn the
    relationship between these factors and the return over the next 63 days (one quarter).
3.  In the out-of-sample backtest (the final 30% of the data), the strategy uses
    the trained model to predict the next quarter's return at each step.
4.  If the predicted return is positive, it enters a long position.
5.  If the predicted return is negative, it enters a short position.
6.  Positions are held for 63 days, as per the original strategy's holding period.
7.  The strategy also adheres to the mandatory development guidelines, including an
    ATR-based stop loss and a higher-timeframe trend filter.
"""
import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest
from sklearn.linear_model import LinearRegression
import warnings

# Suppress potential warnings from backtesting.py
warnings.filterwarnings('ignore', category=UserWarning)


def preprocess_data(df, **params):
    """
    Adds technical indicators as factors, the future return as the target variable,
    and a higher-timeframe trend filter.
    """
    # 1. Technical Factors
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macdsignal
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
    df['BB_upper'] = upper
    df['BB_lower'] = lower
    df['BB_width'] = (upper - lower) / middle

    # 2. Higher-Timeframe Trend Filter (as per guidelines)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['EMA200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_trend_up'] = (df_4h['Close'] > df_4h['EMA200']).astype(int)

    # Map the 4H trend back to the 15m timeframe
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')

    # 3. Target Variable: Future Quarterly Return
    # 63 trading days * 24 hours/day * 4 quarters/hour = 6048 15-min bars
    holding_period = 63 * 24 * 4
    df['future_return'] = df['Close'].shift(-holding_period) / df['Close'] - 1

    # 4. Data Cleaning
    # Drop rows with NaNs created by indicators and the future_return shift
    df.dropna(inplace=True)

    return df


class CrossSectionalProxyStrategy(Strategy):
    """
    A proxy strategy that uses a linear regression model on technical factors
    to predict future returns.
    """

    def init(self):
        """
        This will be implemented in a later step to train the model.
        """
        pass

    def next(self):
        """
        This will be implemented in a later step to generate signals and trade.
        """
        pass


# Main execution block
if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure the data file is in the correct location.")
        exit()

    # Preprocess the data (will be fully implemented later)
    df_processed = preprocess_data(df.copy())

    # Drop any NaN values that might be present after preprocessing
    df_processed.dropna(inplace=True)

    print("Data loaded and preprocessed (initial).")
    print("DataFrame shape:", df_processed.shape)

    if df_processed.empty:
        print("DataFrame is empty after preprocessing and dropping NaNs. Cannot proceed.")
    else:
        # Run backtest
        bt = Backtest(df_processed, CrossSectionalProxyStrategy, cash=100_000, commission=.002)

        # We will run stats and plots in a later step once the strategy is implemented.
        print("Initial backtest object created. Full run will be performed in later steps.")
        # stats = bt.run()
        # print(stats)
        # bt.plot(filename='results/cross_sectional_proxy_plot.html')
