from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import talib

def preprocess_data(df, htf_ema_period=200, volume_ma_period=20, atr_period=14):
    """
    Adds all necessary indicators to the DataFrame for the strategy.
    """
    # 4H Trend Filter
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['ema'] = talib.EMA(df_4h['Close'], timeperiod=htf_ema_period)
    df_4h['htf_trend_up'] = (df_4h['Close'] > df_4h['ema']).astype(float)

    # 4H Predictable Candle Signal
    df_4h['prev_body_high'] = df_4h[['Open', 'Close']].max(axis=1).shift(1)
    df_4h['prev_body_low'] = df_4h[['Open', 'Close']].min(axis=1).shift(1)
    df_4h['body_high'] = df_4h[['Open', 'Close']].max(axis=1)
    df_4h['body_low'] = df_4h[['Open', 'Close']].min(axis=1)

    predict_bullish_series = (df_4h['body_high'] > df_4h['prev_body_high']).astype(float)
    predict_bearish_series = (df_4h['body_low'] < df_4h['prev_body_low']).astype(float)

    # Map HTF signals to the 15m DataFrame
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')

    # Create signal columns for the first bar of the 4H candle
    df['predict_bullish_raw'] = predict_bullish_series.reindex(df.index, method='ffill')
    df['predict_bearish_raw'] = predict_bearish_series.reindex(df.index, method='ffill')

    df['is_new_4h'] = df.index.floor('4H') != df.index.floor('4H').shift(1)

    df['predict_bullish'] = df['is_new_4h'] & (df['predict_bullish_raw'] == 1)
    df['predict_bearish'] = df['is_new_4h'] & (df['predict_bearish_raw'] == 1)

    # Clean up intermediate columns
    df.drop(columns=['predict_bullish_raw', 'predict_bearish_raw', 'is_new_4h'], inplace=True)

    # 15m Indicators
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_period)
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=volume_ma_period)

    return df

class OneCandleHtfMitigationClose(Strategy):
    """
    This strategy identifies a "predictable" 4H candle after it interacts with a higher time frame (HTF)
    key level and confirms its intended direction with a specific body close relative to the previous candle.
    The entry occurs on a lower timeframe (LTF) during the initial manipulation (open to high/low)
    of this predictable 4H candle.
    """

    # --- Optimizable Parameters ---
    atr_period = 14
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    htf_ema_period = 200
    volume_ma_period = 20

    def init(self):
        """
        Initialize indicators using self.I()
        """
        # For simplicity in accessing data, we'll use direct dataframe access within next()
        # but keep the I() mapping for potential plotting or stat extensions.
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name="htf_trend_up")
        self.predict_bullish = self.I(lambda: self.data.predict_bullish, name="predict_bullish")
        self.predict_bearish = self.I(lambda: self.data.predict_bearish, name="predict_bearish")
        self.atr = self.I(lambda: self.data.atr, name="atr")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="volume_ma")

    def next(self):
        """
        Main trading logic.
        """
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]

        # --- Filters ---
        # 1. Volume Filter: Must be above the moving average
        if volume < self.volume_ma[-1]:
            return

        # --- Entry Logic ---
        if not self.position:
            # 2. Bullish Entry
            # Condition: HTF trend is up AND a bullish predictable candle signal occurs
            if self.htf_trend_up[-1] and self.predict_bullish[-1]:
                sl = price - self.atr[-1] * self.atr_sl_multiplier
                tp = price + self.atr[-1] * self.atr_tp_multiplier
                self.buy(sl=sl, tp=tp)

            # 3. Bearish Entry
            # Condition: HTF trend is down AND a bearish predictable candle signal occurs
            elif not self.htf_trend_up[-1] and self.predict_bearish[-1]:
                sl = price + self.atr[-1] * self.atr_sl_multiplier
                tp = price - self.atr[-1] * self.atr_tp_multiplier
                self.sell(sl=sl, tp=tp)

# --- Standalone Execution ---
if __name__ == '__main__':
    import os
    import json

    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    plot_filename = os.path.join(results_dir, 'one_candle_htf_mitigation_close.html')
    json_filename = os.path.join(results_dir, 'temp_result.json')

    # --- Create results directory if it doesn't exist ---
    os.makedirs(results_dir, exist_ok=True)

    # --- Load Data ---
    try:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # As a fallback, you could generate synthetic data here
        exit()

    # --- Preprocess Data ---
    data = preprocess_data(data, htf_ema_period=200) # Using default params
    data.dropna(inplace=True)

    # --- Run Backtest ---
    bt = Backtest(data, OneCandleHtfMitigationClose, cash=100_000, commission=.002)
    stats = bt.run()

    # --- Output ---
    print(stats)
    bt.plot(filename=plot_filename, open_browser=False)

    # --- Save results to JSON ---
    results_dict = stats.to_dict()

    # Sanitize the stats object for JSON serialization
    for key, value in results_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            results_dict[key] = str(value)
        elif isinstance(value, pd.Series):
             results_dict[key] = value.to_dict()
        elif isinstance(value, pd.DataFrame):
             results_dict[key] = value.to_dict()


    with open(json_filename, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print(f"\\nResults saved to {json_filename}")
    print(f"Plot saved to {plot_filename}")
