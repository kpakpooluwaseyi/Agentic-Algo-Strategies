import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.lib import resample_apply

def EMA(series, n):
    """Custom EMA function to use with resample_apply."""
    return pd.Series(series).ewm(span=n, min_periods=n).mean()

def PtaBollingerBands(close_series, length=20, std=2.0):
    """Wrapper for pandas_ta.bbands."""
    close_series_pd = pd.Series(close_series)
    bbands = ta.bbands(close=close_series_pd, length=length, std=std)
    if bbands is None or bbands.empty:
        return np.nan, np.nan, np.nan
    # Dynamically find column names to avoid issues with pandas_ta versions
    lower_col = next((col for col in bbands.columns if col.startswith('BBL')), None)
    middle_col = next((col for col in bbands.columns if col.startswith('BBM')), None)
    upper_col = next((col for col in bbands.columns if col.startswith('BBU')), None)
    if not all([lower_col, middle_col, upper_col]):
        return np.nan, np.nan, np.nan
    return bbands[lower_col].values, bbands[middle_col].values, bbands[upper_col].values

def PtaAtr(high, low, close, length=14):
    """Wrapper for pandas_ta.atr."""
    atr = ta.atr(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), length=length)
    if atr is None or atr.empty:
        return np.nan
    return atr.values

def SMA(series, n):
    """Custom SMA function."""
    return pd.Series(series).rolling(n).mean()

def generate_synthetic_data():
    """Generates synthetic data for testing the strategy."""
    print("Generating synthetic data...")
    n_points = 3000
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))
    price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.1)
    price += np.sin(np.linspace(0, 300, n_points)) * 3
    volume = np.random.randint(100, 1000, n_points)
    data = pd.DataFrame({'Open': price, 'High': price * 1.005, 'Low': price * 0.995,
                         'Close': price, 'Volume': volume}, index=index)
    return data

class MeanReversionWithGuidelines(Strategy):
    """
    Mean-reversion strategy compliant with repository guidelines.
    - Trend Filter: Uses a 4-hour EMA to determine the overall trend direction.
    - Entry: Enters on Bollinger Band breakouts in the direction of the trend.
    - Volume Confirmation: Requires entry bar volume to be above its moving average.
    - Risk Management: Uses ATR for dynamic stop-loss and take-profit levels.
    """
    # Bollinger Bands parameters
    bb_period = 20
    bb_std_dev = 2.0
    # ATR parameters
    atr_period = 14
    sl_atr_multiplier = 2.0
    tp_atr_multiplier = 4.0
    # Volume MA period
    volume_ma_period = 20
    # 4-Hour EMA period
    ema_period = 50

    def init(self):
        # Multi-Timeframe Trend Filter (4H EMA)
        self.ema_4h = resample_apply('4H', EMA, self.data.Close, self.ema_period)

        # Bollinger Bands
        self.lower_band, self.middle_band, self.upper_band = self.I(
            PtaBollingerBands, self.data.Close, self.bb_period, self.bb_std_dev
        )
        # ATR for risk management
        self.atr = self.I(PtaAtr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

        # Volume Confirmation
        self.volume_ma = self.I(SMA, self.data.Volume, self.volume_ma_period)

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]

        # Exit logic: Revert to the mean (middle BB band)
        if self.position:
            if self.position.is_long and price >= self.middle_band[-1]:
                self.position.close()
            elif self.position.is_short and price <= self.middle_band[-1]:
                self.position.close()

        # Entry logic
        elif not self.position:
            volume = self.data.Volume[-1]

            # Long entry conditions
            if (price > self.ema_4h[-1] and                      # Trend confirmation
                crossover(self.lower_band, self.data.Close) and  # BB mean-reversion signal
                volume > self.volume_ma[-1]):                    # Volume confirmation

                sl = price - atr_val * self.sl_atr_multiplier
                tp = price + atr_val * self.tp_atr_multiplier
                self.buy(sl=sl, tp=tp)

            # Short entry conditions
            elif (price < self.ema_4h[-1] and                     # Trend confirmation
                  crossover(self.data.Close, self.upper_band) and # BB mean-reversion signal
                  volume > self.volume_ma[-1]):                   # Volume confirmation

                sl = price + atr_val * self.sl_atr_multiplier
                tp = price - atr_val * self.tp_atr_multiplier
                self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        try:
            data = pd.read_csv(
                data_path, index_col='datetime', parse_dates=True, header=0,
                names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unnamed'],
                usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            ).sort_index()
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'.")
        data = generate_synthetic_data()

    bt = Backtest(data, MeanReversionWithGuidelines, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)): continue
            if pd.isna(value) or (isinstance(value, (float, int)) and not np.isfinite(value)):
                sanitized[key] = None
            elif isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value)
            elif isinstance(value, int):
                sanitized[key] = int(value)
            elif isinstance(value, pd.Timestamp):
                sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                sanitized[key] = str(value)
            elif key.startswith('_'):
                continue
            else:
                sanitized[key] = value
        return sanitized

    final_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    try:
        plot_filename = 'results/mean_reversion_with_guidelines.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
