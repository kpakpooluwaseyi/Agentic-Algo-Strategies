
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Strategy, Backtest
from scipy.signal import find_peaks
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df, **params):
    """
    Adds all necessary indicators and filters to the main DataFrame.
    """
    df = df.copy()

    # Default parameters from strategy class if not provided
    ema_period = params.get('ema_period', 200)
    htf_ema_period = params.get('htf_ema_period', 50)
    atr_period = params.get('atr_period', 14)
    volume_ma_period = params.get('volume_ma_period', 20)

    # 1. Trendline Proxy (EMA)
    df['ema'] = ta.ema(df['Close'], length=ema_period)

    # 2. VuManchu Cipher B Indicators
    df = cipher_b(df)

    # 4. ATR for Risk Management
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=atr_period)

    # 5. Higher-Timeframe (HTF) Filter (4H)
    df_4h = df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
    }).dropna()
    df_4h['htf_ema'] = ta.ema(df_4h['Close'], length=htf_ema_period)
    df['htf_trend_up'] = (df_4h['Close'] > df_4h['htf_ema']).reindex(df.index, method='ffill')
    df['htf_trend_up'] = df['htf_trend_up'].fillna(False) # Fix: Avoid ChainedAssignmentError

    # 6. Volume Moving Average
    df['volume_ma'] = df['Volume'].rolling(volume_ma_period).mean()

    return df

class TrendlineReversalEntry(Strategy):
    """
    Strategy based on trendline reversals, incorporating multi-timeframe analysis,
    volume confirmation, and ATR-based risk management, with VuManchu indicators for confluence.
    """
    # Optimizable Parameters
    atr_period = 14
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_period = 20
    ema_period = 200
    htf_ema_period = 50
    proximity_pct = 0.005 # How close price must be to EMA to be considered a "touch"

    def init(self):
        """
        Initialize indicators.
        """
        # Make pre-calculated data accessible to the strategy
        self.ema = self.I(lambda: self.data.ema, name="EMA_Trendline")
        self.atr = self.I(lambda: self.data.atr, name="ATR")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="Volume_MA")
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name="HTF_Trend_Up")
        self.rsimfi = self.I(lambda: self.data.rsimfi, name="RSI_MFI")

    def next(self):
        """
        Main trading logic.
        """
        # Get the current values of indicators
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        is_htf_up = self.htf_trend_up[-1]
        ema_val = self.ema[-1]

        # Condition: position is not open
        if self.position:
            return

        # --- LONG ENTRY ---
        # 1. Higher timeframe trend is up
        if is_htf_up:
            # 2. Price touches or is very close to the EMA (trendline proxy)
            if self.data.Low[-1] <= ema_val * (1 + self.proximity_pct) and price > ema_val:
                # 3. Volume is above its moving average
                if self.data.Volume[-1] > self.volume_ma[-1]:
                    # 4. VuManchu MFI is positive for confluence
                    if self.rsimfi[-1] > 0:
                        # Calculate SL and TP
                        stop_loss = price - (self.atr_sl_multiplier * atr_val)
                        take_profit = price + (self.atr_tp_multiplier * atr_val)

                        # Place the buy order
                        self.buy(sl=stop_loss, tp=take_profit)
                        return

        # --- SHORT ENTRY ---
        # 1. Higher timeframe trend is down
        if not is_htf_up:
            # 2. Price touches or is very close to the EMA (trendline proxy)
            if self.data.High[-1] >= ema_val * (1 - self.proximity_pct) and price < ema_val:
                # 3. Volume is above its moving average
                if self.data.Volume[-1] > self.volume_ma[-1]:
                    # 4. VuManchu MFI is negative for confluence
                    if self.rsimfi[-1] < 0:
                        # Calculate SL and TP
                        stop_loss = price + (self.atr_sl_multiplier * atr_val)
                        take_profit = price - (self.atr_tp_multiplier * atr_val)

                        # Place the sell order
                        self.sell(sl=stop_loss, tp=take_profit)
                        return

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it's JSON-serializable.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.DataFrame, pd.Series, pd.Timestamp, pd.Timedelta)):
            continue
        if isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value) if not np.isnan(value) else None
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    if '_strategy' in sanitized:
        del sanitized['_strategy']
    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        # As a fallback, create some synthetic data
        dates = pd.date_range(start="2023-01-01", periods=5000, freq='15min')
        price = 20000 + np.cumsum(np.random.randn(5000) * 10)
        data = pd.DataFrame({
            'Open': price,
            'High': price + np.random.uniform(0, 20, 5000),
            'Low': price - np.random.uniform(0, 20, 5000),
            'Close': price + np.random.randn(5000) * 5,
            'Volume': np.random.uniform(100, 1000, 5000)
        }, index=dates)
        data.index.name = 'datetime'
    else:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    data.columns = [c.strip().capitalize() for c in data.columns]

    print("Preprocessing data...")
    preprocessed_data = preprocess_data(data.copy())

    if not preprocessed_data.empty:
        bt = Backtest(preprocessed_data, TrendlineReversalEntry, cash=100_000, commission=.002)

        print("Running backtest...")
        stats = bt.run()
        print(stats)

        os.makedirs('results', exist_ok=True)
        plot_filename = 'results/strategy_bc014e843fbf_plot.html'
        json_filename = 'results/temp_result.json'

        try:
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Backtest plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")

        final_stats = sanitize_stats(stats)
        with open(json_filename, 'w') as f:
            json.dump(final_stats, f, indent=4)
        print(f"Backtest stats saved to {json_filename}")
    else:
        print("Preprocessed data is empty. Skipping backtest.")
