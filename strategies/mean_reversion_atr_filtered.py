# STRATEGY: MeanReversionAtrFiltered
# DESCRIPTION: This strategy implements a mean-reversion approach using Bollinger Bands,
# adhering to the project's specified development guidelines. It includes ATR-based risk management,
# a multi-timeframe trend filter, and volume confirmation for entries.
#
# NOTE on Inheritance: The original request specified inheriting from `MoonDevStrategy`,
# which does not exist. This class inherits from `backtesting.Strategy` to align with
# the established convention in this repository.

import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def PtaBollingerBands(close_series, length=20, std=2.0):
    """
    Wrapper for pandas_ta.bbands that returns a tuple of numpy arrays
    as expected by backtesting.py's self.I().
    """
    bbands = ta.bbands(close=pd.Series(close_series), length=length, std=std)
    if bbands is None or bbands.empty:
        return np.nan, np.nan, np.nan

    # Dynamically find columns to handle variations in pandas_ta versions
    try:
        lower_col = [col for col in bbands.columns if col.startswith('BBL_')][0]
        middle_col = [col for col in bbands.columns if col.startswith('BBM_')][0]
        upper_col = [col for col in bbands.columns if col.startswith('BBU_')][0]
        return bbands[lower_col].values, bbands[middle_col].values, bbands[upper_col].values
    except IndexError:
        return np.nan, np.nan, np.nan

def PtaAtr(high, low, close, length=14):
    """Wrapper for pandas_ta.atr to be used with self.I()."""
    atr = ta.atr(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), length=length)
    return atr.values if atr is not None else np.nan

def preprocess_data(df):
    """
    Calculates the 4-hour EMA and merges it into the main DataFrame.
    """
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).copy()

    df_4h['EMA_50'] = ta.ema(df_4h['Close'], length=50)

    # Merge the 4H EMA back into the 15m dataframe
    df = pd.merge(df, df_4h[['EMA_50']], left_index=True, right_index=True, how='left')
    df['EMA_50'] = df['EMA_50'].ffill()
    return df

class MeanReversionAtrFiltered(Strategy):
    """
    A mean-reversion strategy using Bollinger Bands with ATR-based risk management,
    a 4-hour EMA trend filter, and volume confirmation.
    """
    bb_period = 20
    bb_std_dev = 2.0
    atr_period = 14
    sl_multiplier = 2.0
    tp_multiplier = 3.0
    volume_ma_period = 20

    def init(self):
        # Bollinger Bands
        self.lower_band, self.middle_band, self.upper_band = self.I(
            PtaBollingerBands, self.data.Close, self.bb_period, self.bb_std_dev
        )

        # ATR for risk management
        self.atr = self.I(
            PtaAtr, self.data.High, self.data.Low, self.data.Close, self.atr_period
        )

        # Volume MA for confirmation
        self.volume_ma = self.I(
            lambda x: pd.Series(x).rolling(self.volume_ma_period).mean(),
            self.data.Volume,
            plot=False
        )

        # Access the pre-calculated 4-hour EMA
        self.ema_4h = self.I(lambda: self.data.EMA_50, name='4H_EMA')

    def next(self):
        price = self.data.Close[-1]

        # Trend filter: Only trade in the direction of the 4H trend
        long_trend = price > self.ema_4h[-1]
        short_trend = price < self.ema_4h[-1]

        # Volume filter
        volume_ok = self.data.Volume[-1] > self.volume_ma[-1]

        # --- Entry Conditions ---
        if not self.position:
            # Long entry: price crosses below lower BB, trend is up, volume confirms
            if long_trend and volume_ok and crossover(self.lower_band, self.data.Close):
                sl = price - self.atr[-1] * self.sl_multiplier
                tp = price + self.atr[-1] * self.tp_multiplier
                self.buy(sl=sl, tp=tp)

            # Short entry: price crosses above upper BB, trend is down, volume confirms
            elif short_trend and volume_ok and crossover(self.data.Close, self.upper_band):
                sl = price + self.atr[-1] * self.sl_multiplier
                tp = price - self.atr[-1] * self.tp_multiplier
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        try:
            # Load data, ensuring correct headers and parsing dates.
            data = pd.read_csv(
                data_path,
                index_col='datetime',
                parse_dates=True,
                header=0,
                names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unnamed'],
                usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'.")
        data = generate_synthetic_data()

    # Preprocess the data
    data = preprocess_data(data)

    bt = Backtest(data, MeanReversionAtrFiltered, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)): continue
            if isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, int): sanitized[key] = int(value)
            elif isinstance(value, pd.Timestamp): sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta): sanitized[key] = str(value)
            elif pd.isna(value): sanitized[key] = None
            elif key.startswith('_'): continue
            else: sanitized[key] = value
        return sanitized

    final_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    try:
        plot_filename = 'results/mean_reversion_atr_filtered.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
