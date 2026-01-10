
import backtesting as bt
import numpy as np
import pandas as pd
import pandas_ta as ta
import json
import os

def sanitize_stats(stats):
    """Sanitizes the stats object for JSON serialization."""
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, pd.DataFrame):
            # Handle DataFrames separately to avoid the truth value error
            if not value.empty:
                df_copy = value.copy()
                for col in df_copy.select_dtypes(include=['datetime64[ns]', 'timedelta64[ns]']).columns:
                    df_copy[col] = df_copy[col].astype(str).replace('NaT', None)
                sanitized[key] = df_copy.to_dict(orient='records')
            else:
                sanitized[key] = []
        elif value is pd.NaT or pd.isna(value):
            sanitized[key] = None
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, (int, float, str, bool, type(None))):
            sanitized[key] = value
        elif isinstance(value, pd.DataFrame):
            if not value.empty:
                df_copy = value.copy()
                for col in df_copy.select_dtypes(include=['datetime64[ns]', 'timedelta64[ns]']).columns:
                    df_copy[col] = df_copy[col].astype(str).replace('NaT', None)
                sanitized[key] = df_copy.to_dict(orient='records')
            else:
                sanitized[key] = []
        else:
            sanitized[key] = str(value)
    if '_strategy' in sanitized:
        sanitized['_strategy'] = str(sanitized['_strategy'])
    return sanitized

def preprocess_data(df):
    """Applies indicators and filters to the dataframe."""
    # Main timeframe indicators
    df['sma'] = ta.sma(df['Close'], length=50)
    df['volume_ma'] = ta.sma(df['Volume'], length=20)
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Log prices
    df['log_close'] = np.log(df['Close'])
    df['log_sma'] = np.log(df['sma'])

    # 4H timeframe filter
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['trend_ma'] = ta.sma(df_4h['Close'], length=50)
    df_4h['uptrend'] = df_4h['Close'] > df_4h['trend_ma']

    # Map 4H trend to 15m data
    df['4h_uptrend'] = df.index.floor('4H').map(df_4h['uptrend'])
    df['4h_uptrend'].fillna(method='ffill', inplace=True)

    df.dropna(inplace=True)
    return df

class LongOnlyMovingAverageBreakout(bt.Strategy):
    ma_period = 50
    breakout_factor = 1.001
    atr_multiplier_sl = 2
    atr_multiplier_tp = 3.5

    def init(self):
        # Make indicators available in the strategy
        self.log_close = self.I(lambda: self.data.df['log_close'])
        self.log_sma = self.I(lambda: self.data.df['log_sma'])
        self.volume = self.I(lambda: self.data.df['Volume'])
        self.volume_ma = self.I(lambda: self.data.df['volume_ma'])
        self.atr_val = self.I(lambda: self.data.df['atr'])
        self.uptrend_4h = self.I(lambda: self.data.df['4h_uptrend'])

    def next(self):
        price = self.data.Close[-1]

        # Entry conditions
        if not self.position:
            # Corrected entry logic
            price_breakout = self.log_close[-1] > (self.log_sma[-1] + np.log(self.breakout_factor))
            volume_confirmed = self.volume[-1] > self.volume_ma[-1]
            trend_confirmed = self.uptrend_4h[-1] == 1

            if price_breakout and volume_confirmed and trend_confirmed:
                sl = price - self.atr_val[-1] * self.atr_multiplier_sl
                tp = price + self.atr_val[-1] * self.atr_multiplier_tp

                if sl > 0 and tp > price:
                    self.buy(sl=sl, tp=tp)

        # Exit condition
        elif self.position.is_long:
            if self.log_close[-1] < self.log_sma[-1]:
                self.position.close()

if __name__ == '__main__':
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv')
        data.columns = [col.strip().capitalize() for col in data.columns]
        data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.set_index('Datetime')
        data = data.sort_index()
    except FileNotFoundError:
        print("Data file not found. Generating sample data.")
        # Sample data generation remains for standalone testing
        periods = 5000
        dates = pd.date_range('2022-01-01', periods=periods, freq='15min')
        np.random.seed(42)
        close = 20000 + (np.random.randn(periods).cumsum() * 2)
        high = close + np.random.uniform(0, 20, size=periods)
        low = close - np.random.uniform(0, 20, size=periods)
        open_ = close + np.random.normal(0, 5, size=periods)
        volume = np.random.randint(100, 5000, size=periods)
        data = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=dates)

    # Preprocess the data
    data = preprocess_data(data)

    bt_instance = bt.Backtest(data, LongOnlyMovingAverageBreakout, cash=100_000, commission=.002)
    stats = bt_instance.run()

    print(stats)

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    sanitized_stats = sanitize_stats(stats)
    with open(os.path.join(results_dir, 'temp_result.json'), 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    plot_filename = os.path.join(results_dir, 'long_only_moving_average_breakout.html')
    bt_instance.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")
