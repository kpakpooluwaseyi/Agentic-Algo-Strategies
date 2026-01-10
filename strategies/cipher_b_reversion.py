
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover
import os
import sys

# Add parent directory to path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.indicators.vumanchu import cipher_b
import pandas_ta as ta

def preprocess_data(df, trend_period=200, volume_period=20, atr_period=14):
    """
    Adds the required indicators to the DataFrame for the CipherBReversion strategy.

    - VuManchu Cipher B signals
    - Long-term EMA for trend filtering
    - Rolling volume average for confirmation
    - ATR for risk management
    """
    # 1. Add Cipher B indicator suite
    df = cipher_b(df)

    # 2. Add long-term EMA for trend filter
    df['ema_trend'] = ta.ema(df['Close'], length=trend_period)

    # 3. Add rolling volume average for confirmation
    df['volume_avg'] = ta.sma(df['Volume'], length=volume_period)

    # 4. Add ATR for risk management
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=atr_period)

    # Convert boolean signals to int for backtesting.py compatibility
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    df.dropna(inplace=True)
    return df

class CipherBReversion(Strategy):
    """
    A mean-reversion strategy based on the VuManchu Cipher B indicator
    with trend and volume confirmation, and ATR-based risk management.

    Long Entry:
    - Cipher B buy signal triggers.
    - Price is above the long-term trend-filtering EMA.
    - Volume is above its recent average.

    Short Entry:
    - Cipher B sell signal triggers.
    - Price is below the long-term trend-filtering EMA.
    - Volume is above its recent average.

    Exits:
    - Stop-loss and take-profit levels are set based on a multiple of ATR.
    """

    # --- Optimizable parameters ---
    # Risk Management
    sl_multiplier = 2.0  # ATR multiple for stop-loss
    tp_multiplier = 3.0  # ATR multiple for take-profit

    # Confirmation Filters
    trend_period = 200
    volume_period = 20
    atr_period = 14

    def init(self):
        """
        Initialize the indicators.
        """
        # Signals from preprocess_data
        self.buy_sig = self.I(lambda: self.data.buy_signal, name="buy_signal")
        self.sell_sig = self.I(lambda: self.data.sell_signal, name="sell_signal")

        # Confirmation indicators
        self.ema_trend = self.I(lambda: self.data.ema_trend, name="ema_trend")
        self.volume_avg = self.I(lambda: self.data.volume_avg, name="volume_avg")
        self.atr = self.I(lambda: self.data.atr, name="atr")

    def next(self):
        """
        Define the trading logic.
        """
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        atr_value = self.atr[-1]

        # --- Entry Conditions ---
        if not self.position:
            # Long Entry: Buy signal + Above Trend EMA + High Volume
            if self.buy_sig[-1] == 1 and price > self.ema_trend[-1] and volume > self.volume_avg[-1]:
                sl = price - self.sl_multiplier * atr_value
                tp = price + self.tp_multiplier * atr_value
                self.buy(sl=sl, tp=tp)

            # Short Entry: Sell signal + Below Trend EMA + High Volume
            elif self.sell_sig[-1] == 1 and price < self.ema_trend[-1] and volume > self.volume_avg[-1]:
                sl = price + self.sl_multiplier * atr_value
                tp = price - self.tp_multiplier * atr_value
                self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    from backtesting import Backtest
    import numpy as np
    import json
    import os

    # --- Configuration ---
    DATA_PATH = 'data/BTC-USD-15m.csv'
    STRATEGY = CipherBReversion
    CASH = 100_000
    COMMISSION = .002

    # --- Data Loading ---
    try:
        # Load data with correct column names for backtesting.py
        data = pd.read_csv(DATA_PATH)
        # Sanitize column names
        data.columns = [col.strip().capitalize() for col in data.columns]
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.set_index('Datetime')
        print(f"Loaded {len(data)} rows from {DATA_PATH}")
    except FileNotFoundError:
        print(f"Data file not found at {DATA_PATH}. Generating synthetic data.")
        # Generate synthetic data if the file is not available
        dates = pd.date_range(start='2020-01-01', periods=20000, freq='15min')
        price = 10000 + np.cumsum(np.random.randn(20000)) * 2
        data = pd.DataFrame({
            'Open': price,
            'High': price + np.random.uniform(0, 5, 20000),
            'Low': price - np.random.uniform(0, 5, 20000),
            'Close': price + np.random.randn(20000),
            'Volume': np.random.randint(100, 5000, 20000)
        }, index=dates)
        data.index.name = 'Datetime'


    # --- Preprocessing ---
    data = preprocess_data(data)

    # --- Backtesting ---
    bt = Backtest(data, STRATEGY, cash=CASH, commission=COMMISSION)
    stats = bt.run()
    print("\n--- Backtest Results ---")
    print(stats)

    # --- Save Results ---
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON output
    def sanitize_stats(stats_obj):
        sanitized = {}
        for key, value in stats_obj.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                sanitized[key] = float(value)
            elif isinstance(value, (pd.DataFrame, pd.Series)):
                # Skip DataFrames/Series like _equity_curve and _trades
                continue
            elif pd.isna(value):
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

    results_path = 'results/temp_result.json'
    sanitized_results = sanitize_stats(stats)
    with open(results_path, 'w') as f:
        json.dump(sanitized_results, f, indent=4)
    print(f"\nResults saved to {results_path}")

    # --- Plotting ---
    plot_path = f"results/{STRATEGY.__name__}.html"
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"\nCould not generate plot due to an error: {e}")
