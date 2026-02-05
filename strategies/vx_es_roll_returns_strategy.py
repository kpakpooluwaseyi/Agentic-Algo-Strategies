
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy

def preprocess_data(df, short_atr_period=14, long_atr_period=100, atr_period=14, mtf_ema_period=50, volume_period=30):
    """
    Calculates indicators for the VxEsRollReturnsStrategy proxy, including MTF and volume filters.
    """
    # Base indicators
    df['atr_short'] = ta.atr(df.High, df.Low, df.Close, length=short_atr_period)
    df['atr_long'] = ta.atr(df.High, df.Low, df.Close, length=long_atr_period)
    df['volatility_ratio'] = df['atr_short'] / df['atr_long']
    df['risk_atr'] = ta.atr(df.High, df.Low, df.Close, length=atr_period)

    # Multi-Timeframe (MTF) Filter: 4-hour EMA
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['ema_4h'] = ta.ema(df_4h.Close, length=mtf_ema_period)
    df_4h['trend_4h'] = np.where(df_4h.Close > df_4h.ema_4h, 1, -1)
    df['trend_4h'] = df_4h['trend_4h'].reindex(df.index, method='ffill')

    # Volume Confirmation
    df['volume_avg'] = ta.sma(df.Volume, length=volume_period)

    df.dropna(inplace=True)
    return df

class VxEsRollReturnsStrategy(Strategy):
    """
    This is a proxy implementation of the "VX/ES Roll Returns Strategy".
    It adapts the core concept for a single spot asset (e.g., Bitcoin) by using a
    volatility ratio to model the VIX term structure. It also includes
    mandatory MTF and volume filters as per development guidelines.

    NOTE: The request specified inheriting from `MoonDevStrategy`, but that class
    belongs to an incompatible internal framework. This strategy inherits from
    `backtesting.Strategy` to align with all other functional strategies in this repository.
    """
    short_atr_period = 14
    long_atr_period = 100
    atr_period = 14
    mtf_ema_period = 50
    volume_period = 30

    long_threshold = 1.2
    short_threshold = 0.8

    atr_stop_loss_multiplier = 2.0
    atr_take_profit_multiplier = 3.0

    def init(self):
        self.volatility_ratio = self.data.volatility_ratio
        self.risk_atr = self.data.risk_atr
        self.trend_4h = self.data.trend_4h
        self.volume_avg = self.data.volume_avg

    def next(self):
        if self.position:
            return

        current_vol_ratio = self.volatility_ratio[-1]

        # Long Entry: High volatility + Bullish confirmation + 4H Uptrend + High Volume
        if current_vol_ratio >= self.long_threshold and self.trend_4h[-1] == 1 and self.data.Volume[-1] > self.volume_avg[-1]:
            if self.data.Close[-1] > self.data.Open[-1]:
                atr_value = self.risk_atr[-1]
                sl = self.data.Close[-1] - self.atr_stop_loss_multiplier * atr_value
                tp = self.data.Close[-1] + self.atr_take_profit_multiplier * atr_value
                self.buy(sl=sl, tp=tp)

        # Short Entry: Low volatility + Bearish confirmation + 4H Downtrend + High Volume
        elif current_vol_ratio <= self.short_threshold and self.trend_4h[-1] == -1 and self.data.Volume[-1] > self.volume_avg[-1]:
            if self.data.Close[-1] < self.data.Open[-1]:
                atr_value = self.risk_atr[-1]
                sl = self.data.Close[-1] + self.atr_stop_loss_multiplier * atr_value
                tp = self.data.Close[-1] - self.atr_take_profit_multiplier * atr_value
                self.sell(sl=sl, tp=tp)

def sanitize_stats(stats):
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Series, pd.DataFrame)) or key.startswith('_'): continue
        if isinstance(value, (np.floating, np.integer)):
            sanitized[key] = float(value) if np.isfinite(value) else None
        elif isinstance(value, (int, float)): sanitized[key] = value
        elif isinstance(value, pd.Timestamp): sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta): sanitized[key] = str(value)
        elif pd.isna(value): sanitized[key] = None
        else: sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        column_names = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = pd.read_csv(
            data_path, index_col='datetime', parse_dates=True, header=0,
            names=column_names, usecols=column_names
        )
    else:
        n_points = 5000
        index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))
        price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.2)
        data = pd.DataFrame({'Open': price, 'High': price*1.01, 'Low': price*0.99,
                             'Close': price, 'Volume': np.random.randint(100, 2000, n_points)},
                            index=index)

    data = preprocess_data(data)

    bt = Backtest(data, VxEsRollReturnsStrategy, cash=100_000, commission=.002)
    stats = bt.run()

    os.makedirs('results', exist_ok=True)
    final_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    try:
        plot_filename = 'results/vx_es_roll_returns_strategy.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
