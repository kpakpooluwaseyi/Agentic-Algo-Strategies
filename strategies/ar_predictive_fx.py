from backtesting import Strategy
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import talib
import warnings

# Suppress warnings from statsmodels
warnings.filterwarnings("ignore")

def preprocess_data(df, p=10, window=100, **params):
    """
    Preprocesses the data by adding indicators and AR(p) model predictions.

    Args:
        df: DataFrame with OHLCV data.
        p (int): Lag parameter for the AR model.
        window (int): Rolling window size for AR model fitting.
        **params: Additional parameters.

    Returns:
        df: DataFrame with added indicators and predictions.
    """
    df = df.copy()

    # AR(p) Model Predictions
    def ar_predict(series):
        try:
            model = AutoReg(series, lags=p, old_names=False).fit()
            return model.predict(start=len(series), end=len(series)).iloc[0]
        except Exception:
            return np.nan

    if len(df) > window:
        df['predicted_close'] = df['Close'].rolling(window=window).apply(ar_predict, raw=False)
        df['predicted_close'] = df['predicted_close'].shift(1)
    else:
        df['predicted_close'] = np.nan

    # Development Guidelines Indicators
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    df['volume_ma'] = df['Volume'].rolling(20).mean()

    # Higher Timeframe Trend Filter (4H EMA 200)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], 200)
    # Use a boolean for the trend direction
    df['htf_uptrend'] = (df_4h['Close'] > df_4h['ema_200']).reindex(df.index, method='ffill')

    return df

class ARPredictiveFX(Strategy):
    """
    Mean-reversion strategy using an AR(p) model with mandatory development guidelines.
    """
    p_lag = 10
    rolling_window = 100
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        self.predicted_close = self.I(lambda: self.data.predicted_close, name="predicted_close")
        self.atr = self.I(lambda: self.data.atr, name="atr")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="volume_ma")
        # Ensure the htf_uptrend is treated as a numerical series for plotting
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend.astype(float), name="htf_uptrend")

    def next(self):
        # Close position after one bar (as per original strategy spec)
        if self.position:
            self.position.close()
            # Return after closing to ensure we don't re-enter on the same bar
            return

        # --- Guideline Filters ---
        # 1. Volume Confirmation
        if self.data.Volume[-1] < self.volume_ma[-1]:
            return

        # 2. Check for NaN indicators
        if np.isnan(self.predicted_close[-1]) or np.isnan(self.atr[-1]) or pd.isna(self.htf_uptrend[-1]):
            return

        # --- Entry Logic ---
        predicted_price = self.predicted_close[-1]
        current_price = self.data.Close[-1]
        atr_val = self.atr[-1]

        # Long Entry (with HTF trend confirmation)
        if predicted_price > current_price and self.htf_uptrend[-1] == 1:
            sl = current_price - (self.atr_sl_multiplier * atr_val)
            tp = current_price + (self.atr_tp_multiplier * atr_val)
            self.buy(sl=sl, tp=tp)

        # Short Entry (with HTF trend confirmation)
        elif predicted_price < current_price and self.htf_uptrend[-1] == 0:
            sl = current_price + (self.atr_sl_multiplier * atr_val)
            tp = current_price - (self.atr_tp_multiplier * atr_val)
            self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    from backtesting import Backtest

    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
        # Use a larger slice to ensure enough data for 4H resample and indicator warmup
        df = df.iloc[-3000:]
    except FileNotFoundError:
        print("No data file found. Generating sample data...")
        dates = pd.date_range('2023-01-01', periods=3000, freq='15m')
        np.random.seed(42)
        price = 20000 + np.cumsum(np.random.randn(3000) * 10)
        df = pd.DataFrame({
            'Open': price, 'High': price + np.random.rand(3000) * 20,
            'Low': price - np.random.rand(3000) * 20, 'Close': price + np.random.randn(3000) * 5,
            'Volume': np.random.rand(3000) * 1000000
        }, index=dates)

    print("Preprocessing data... This may take a while.")
    df = preprocess_data(df, p=ARPredictiveFX.p_lag, window=ARPredictiveFX.rolling_window)
    df.dropna(inplace=True)

    if df.empty:
        print("DataFrame is empty after preprocessing. Check data and parameters.")
    else:
        bt = Backtest(df, ARPredictiveFX, cash=100000, commission=0.001)
        stats = bt.run()
        print(stats)
        bt.plot(filename='results/ar_predictive_fx_final.html')
