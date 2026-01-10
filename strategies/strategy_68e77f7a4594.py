
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
import pandas as pd
import talib

def preprocess_data(df: pd.DataFrame, fast_ema_period=20, slow_ema_period=50, trend_ema_period=200, atr_period=14, volume_ma_period=20):
    """
    Adds all required indicators to the DataFrame.
    """
    df['fast_ema'] = talib.EMA(df['Close'], timeperiod=fast_ema_period)
    df['slow_ema'] = talib.EMA(df['Close'], timeperiod=slow_ema_period)
    df['trend_ema'] = talib.EMA(df['Close'], timeperiod=trend_ema_period)

    # 4-hour trend filter
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['trend_ema'] = talib.EMA(df_4h['Close'], timeperiod=trend_ema_period)
    df_4h['htf_uptrend'] = df_4h['Close'] > df_4h['trend_ema']
    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill').fillna(False)

    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_period)
    df['volume_ma'] = df['Volume'].rolling(volume_ma_period).mean()

    return df

class EmaCrossoverFiltered(Strategy):
    fast_ema_period = 20
    slow_ema_period = 50
    trend_ema_period = 200
    atr_period = 14
    volume_ma_period = 20
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        self.fast_ema = self.I(lambda: self.data.fast_ema, name='fast_ema')
        self.slow_ema = self.I(lambda: self.data.slow_ema, name='slow_ema')
        self.trend_ema = self.I(lambda: self.data.trend_ema, name='trend_ema')
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend, name='htf_uptrend')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')

    def next(self):
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]

        # Filter conditions
        is_htf_uptrend = self.htf_uptrend[-1]
        is_volume_above_average = volume > self.volume_ma[-1]

        # Entry conditions
        long_signal = crossover(self.fast_ema, self.slow_ema) and self.slow_ema[-1] > self.trend_ema[-1]
        short_signal = crossover(self.slow_ema, self.fast_ema) and self.slow_ema[-1] < self.trend_ema[-1]

        if not self.position:
            # Long entry logic
            if long_signal and is_htf_uptrend and is_volume_above_average:
                sl = price - self.atr[-1] * self.atr_sl_multiplier
                tp = price + self.atr[-1] * self.atr_tp_multiplier
                self.buy(sl=sl, tp=tp)

            # Short entry logic
            elif short_signal and not is_htf_uptrend and is_volume_above_average:
                sl = price + self.atr[-1] * self.atr_sl_multiplier
                tp = price - self.atr[-1] * self.atr_tp_multiplier
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    import os
    import json

    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    # Sanitize column names
    df.columns = [col.strip().title() for col in df.columns]

    # Preprocess the data
    processed_df = preprocess_data(df.copy())

    bt = Backtest(processed_df, EmaCrossoverFiltered, cash=100_000, commission=.002)
    stats = bt.run()

    print(stats)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    bt.plot(filename='results/ema_crossover_filtered.html')

    # Save stats to json
    stats_dict = stats.to_dict()

    # Remove non-serializable items
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    # Sanitize remaining stats for JSON serialization
    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)
        elif pd.isna(value):
            stats_dict[key] = None
        elif hasattr(value, 'item'): # Convert numpy types to python types
            stats_dict[key] = value.item()

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)
