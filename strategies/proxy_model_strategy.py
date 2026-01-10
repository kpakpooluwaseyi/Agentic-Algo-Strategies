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
    df.dropna(inplace=True)

    return df


class CrossSectionalProxyStrategy(Strategy):
    """
    A proxy strategy that uses a linear regression model on technical factors
    to predict future returns.
    """
    train_pct = 0.7
    holding_period = 63 * 24 * 4
    atr_sl_multiplier = 2.0

    def init(self):
        self.atr = self.I(lambda: self.data.ATR, name='ATR')
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name='htf_trend_up')

        self.factor_columns = [
            'RSI', 'ADX', 'ATR', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower', 'BB_width'
        ]

        split_index = int(len(self.data.Close) * self.train_pct)

        if split_index < 2:  # Need at least 2 samples to train
            self.model = None
            return

        train_df = self.data.df.iloc[:split_index]
        X_train = train_df[self.factor_columns]
        y_train = train_df['future_return']

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.start_oos_index = split_index

    def next(self):
        if self.model is None or len(self.data.Close) < self.start_oos_index:
            return

        if self.position:
            if len(self.data.Close) - self.trades[0].entry_bar >= self.holding_period:
                self.position.close()
            return # Don't check for new entries while in a position

        # HTF Filter: Only take longs if 4H trend is up. Shorts can be taken anytime.
        if self.htf_trend_up[-1] == 0 and self.model.predict(self.data.df.iloc[-1][self.factor_columns].values.reshape(1, -1))[0] > 0:
            return

        current_factors = self.data.df.iloc[-1][self.factor_columns].values.reshape(1, -1)
        predicted_return = self.model.predict(current_factors)[0]

        if predicted_return > 0:
            sl = self.data.Close[-1] - self.atr[-1] * self.atr_sl_multiplier
            self.buy(sl=sl)
        elif predicted_return < 0:
            sl = self.data.Close[-1] + self.atr[-1] * self.atr_sl_multiplier
            self.sell(sl=sl)


import json

def sanitize_stats(stats):
    """
    Sanitizes the backtesting stats object by converting non-serializable
    types to JSON-compatible types. Re-ordered to check for complex types first.
    """
    if isinstance(stats, pd.Series):
        stats = stats.to_dict()

    clean_stats = {}
    for key, value in stats.items():
        # First, check for and skip complex, non-serializable objects
        if isinstance(value, (pd.DataFrame, pd.Series, Strategy, type(Backtest))):
            continue
        # Now, handle conversions for serializable types
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            clean_stats[key] = str(value)
        elif pd.isna(value):
            clean_stats[key] = None
        elif isinstance(value, (np.integer, np.int64)):
            clean_stats[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            clean_stats[key] = float(value)
        else:
            clean_stats[key] = value

    # Pop any remaining complex keys just in case
    clean_stats.pop('_strategy', None)
    clean_stats.pop('_equity_curve', None)
    clean_stats.pop('_trades', None)

    return clean_stats

if __name__ == '__main__':
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found.")
        exit()

    # Sanitize column names (e.g., 'open' -> 'Open', ' high ' -> 'High')
    df.columns = [col.strip().capitalize() for col in df.columns]

    # The CSV has a trailing comma, which creates an unnamed, empty column. Remove it.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    df_processed = preprocess_data(df.copy())

    if df_processed.empty:
        print("DataFrame is empty after preprocessing. Cannot run backtest.")
    else:
        print("Running backtest...")
        bt = Backtest(df_processed, CrossSectionalProxyStrategy, cash=100_000, commission=.002)
        stats = bt.run()

        print("\n--- Backtest Results ---")
        print(stats)

        # Save plot
        plot_filename = 'results/proxy_model_strategy.html'
        print(f"\nSaving plot to {plot_filename}...")
        bt.plot(filename=plot_filename, open_browser=False)

        # Save stats to JSON
        stats_filename = 'results/temp_result.json'
        print(f"Saving stats to {stats_filename}...")

        # Sanitize the stats object before saving
        cleaned_stats = sanitize_stats(stats)

        with open(stats_filename, 'w') as f:
            json.dump(cleaned_stats, f, indent=4)

        print("\nBacktest complete.")
