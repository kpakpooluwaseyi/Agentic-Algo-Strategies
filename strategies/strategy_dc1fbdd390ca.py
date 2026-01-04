
from backtesting import Strategy, Backtest
import pandas as pd
import talib
import numpy as np

def preprocess_data(df, **params):
    """
    Adds all necessary indicators to the DataFrame for the EmaCloudTrendFollowing strategy.

    - EMA Fast (20) and Slow (50) for the primary timeframe cloud.
    - 4-hour timeframe trend filter using a 200-period EMA.
    - ATR (14) for dynamic risk management (Stop Loss and Take Profit).
    - Volume Moving Average (20) for entry confirmation.
    """
    df = df.copy()

    # Get parameters or use defaults
    ema_fast_period = params.get('ema_fast_period', 20)
    ema_slow_period = params.get('ema_slow_period', 50)
    htf_ema_period = params.get('htf_ema_period', 200)
    atr_period = params.get('atr_period', 14)
    volume_ma_period = params.get('volume_ma_period', 20)

    # Primary timeframe EMAs
    df['ema_fast'] = talib.EMA(df['Close'], timeperiod=ema_fast_period)
    df['ema_slow'] = talib.EMA(df['Close'], timeperiod=ema_slow_period)

    # Higher timeframe (4H) trend filter
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['htf_ema'] = talib.EMA(df_4h['Close'], timeperiod=htf_ema_period)
    df_4h['htf_trend_up'] = df_4h['Close'] > df_4h['htf_ema']

    # Merge HTF trend back to the main DataFrame
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')
    df['htf_trend_up'].fillna(False, inplace=True) # Ensure no NaN at the start

    # ATR for risk management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_period)

    # Volume MA for confirmation
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=volume_ma_period)

    # backtesting.py will handle NaNs, so we don't need to drop them here.
    # df.dropna(inplace=True)

    return df

class EmaCloudTrendFollowing(Strategy):
    """
    Implementation of an EMA Cloud trend-following strategy.

    Entry Rules:
    Long:
      - Higher timeframe (4h) trend is up (Close > EMA200).
      - Primary timeframe (15m) EMA cloud is sloping up and price is above the cloud.
      - Volume is above its moving average.
    Short:
      - Higher timeframe (4h) trend is down (Close < EMA200).
      - Primary timeframe (15m) EMA cloud is sloping down and price is below the cloud.
      - Volume is above its moving average.

    Exit Rules:
    - Stop Loss: ATR-based (2x ATR)
    - Take Profit: ATR-based (3x ATR)
    """

    # Optimizable parameters
    ema_fast_period = 20
    ema_slow_period = 50
    htf_ema_period = 200
    atr_period = 14
    volume_ma_period = 20
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        """
        Initialize indicators from the preprocessed DataFrame.
        """
        self.ema_fast = self.I(lambda: self.data.ema_fast, name="EMA_Fast")
        self.ema_slow = self.I(lambda: self.data.ema_slow, name="EMA_Slow")
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name="HTF_Trend_Up")
        self.atr = self.I(lambda: self.data.atr, name="ATR")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="Volume_MA")

    def next(self):
        """
        Defines the trading logic for each bar.
        """
        price = self.data.Close[-1]

        # --- Trend Conditions ---
        is_htf_up = self.htf_trend_up[-1]

        # Primary timeframe trend conditions
        is_primary_up = price > self.ema_fast[-1] and self.ema_fast[-1] > self.ema_slow[-1]
        is_primary_down = price < self.ema_fast[-1] and self.ema_fast[-1] < self.ema_slow[-1]

        # Volume condition
        volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]

        # --- ENTRY LOGIC ---
        # If no position is open, check for entry signals
        if not self.position:
            # Long entry: HTF uptrend + Primary uptrend + Volume confirmation
            if is_htf_up and is_primary_up and volume_confirmed:
                sl = price - self.atr_sl_multiplier * self.atr[-1]
                tp = price + self.atr_tp_multiplier * self.atr[-1]
                self.buy(sl=sl, tp=tp)

            # Short entry: HTF downtrend + Primary downtrend + Volume confirmation
            elif not is_htf_up and is_primary_down and volume_confirmed:
                sl = price + self.atr_sl_multiplier * self.atr[-1]
                tp = price - self.atr_tp_multiplier * self.atr[-1]
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        # Clean and capitalize column names to prevent KeyErrors
        df.columns = [col.strip().capitalize() for col in df.columns]
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure the data file is in the correct location.")
        exit()

    # Preprocess data
    df_processed = preprocess_data(df)

    # Run backtest
    bt = Backtest(df_processed, EmaCloudTrendFollowing, cash=100_000, commission=.002)
    stats = bt.run()

    # Save results
    import json
    import os

    print("Backtest Stats:")
    print(stats)

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # --- Sanitize and save stats ---
    # Create a new dictionary to store serializable stats
    stats_to_save = {}
    for key, value in stats.items():
        if isinstance(value, (int, float, str, bool)):
            stats_to_save[key] = value
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_to_save[key] = str(value)
        elif isinstance(value, np.number):
            stats_to_save[key] = value.item() # Convert numpy numbers to python types
        # Ignore complex objects like DataFrames or the strategy instance itself

    results_path = os.path.join(results_dir, 'temp_result.json')
    with open(results_path, 'w') as f:
        json.dump(stats_to_save, f, indent=4)
    print(f"Results saved to {results_path}")

    # Generate and save plot
    plot_path = os.path.join(results_dir, 'strategy_dc1fbdd390ca_plot.html')
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
