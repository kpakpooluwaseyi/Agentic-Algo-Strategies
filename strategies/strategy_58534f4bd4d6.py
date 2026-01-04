
import pandas as pd
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
import pandas_ta as ta
import numpy as np
import os
import sys

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Adds all necessary indicators to the DataFrame for the strategy.
    """
    # Add VuManchu Cipher B indicators
    df = cipher_b(df)

    # Add EMAs
    df['ema_fast'] = ta.ema(df.Close, length=params.get('ema_fast_len', 50))
    df['ema_slow'] = ta.ema(df.Close, length=params.get('ema_slow_len', 200))

    # Add ATR for risk management
    df['atr'] = ta.atr(df.High, df.Low, df.Close, length=params.get('atr_len', 14))

    # Add Volume MA for confirmation
    df['volume_ma'] = ta.sma(df.Volume, length=params.get('volume_ma_len', 20))

    # Add higher timeframe (4H) trend filter
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).copy()
    df_4h['ema_200'] = ta.ema(df_4h.Close, length=params.get('ema_slow_len', 200))

    # Map 4H trend to original timeframe
    df['htf_uptrend'] = (df_4h['Close'] > df_4h['ema_200']).reindex(df.index, method='ffill')
    df['htf_uptrend'] = df['htf_uptrend'].fillna(False)

    # Use the original boolean signals from cipher_b and convert to int
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    return df

class VuManchuEMAScalping(Strategy):
    ema_fast_len = 50
    ema_slow_len = 200
    atr_len = 14
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_len = 20
    max_trades_per_cross = 4

    def init(self):
        # Indicators
        self.ema_fast = self.I(lambda: self.data.ema_fast, name="EMA_Fast")
        self.ema_slow = self.I(lambda: self.data.ema_slow, name="EMA_Slow")
        self.atr = self.I(lambda: self.data.atr, name="ATR")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="Volume_MA")
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend, name="HTF_Uptrend")

        # VuManchu signals
        self.vumanchu_buy = self.I(lambda: self.data.buy_signal, name="VuManchu_Buy_Signal")
        self.vumanchu_sell = self.I(lambda: self.data.sell_signal, name="VuManchu_Sell_Signal")
        self.wt1 = self.I(lambda: self.data.wt1, name="WaveTrend1")

        # State tracking
        self.trades_since_cross = 0
        self.last_cross_type = None # "golden" or "death"

    def next(self):
        # Detect crosses and reset trade counter
        is_golden_cross = crossover(self.ema_fast, self.ema_slow)
        is_death_cross = crossover(self.ema_slow, self.ema_fast)

        if is_golden_cross:
            self.last_cross_type = "golden"
            self.trades_since_cross = 0
        elif is_death_cross:
            self.last_cross_type = "death"
            self.trades_since_cross = 0

        # If a trade is open, do not take another
        if self.position:
            return

        # Check if we have exceeded max trades for the current trend
        if self.trades_since_cross >= self.max_trades_per_cross:
            return

        price = self.data.Close[-1]
        atr_value = self.atr[-1]

        # --- Entry Conditions ---
        volume_ok = self.data.Volume[-1] > self.volume_ma[-1]

        # Long Entry Logic
        if self.last_cross_type == "golden":
            is_trade_signal = self.vumanchu_buy[-1] == 1 and self.wt1[-1] < 0
            price_above_slow_ema = price > self.ema_slow[-1]

            if is_trade_signal and self.htf_uptrend[-1] and volume_ok and price_above_slow_ema:
                sl = price - self.atr[-1] * self.atr_sl_multiplier
                tp = price + self.atr[-1] * self.atr_tp_multiplier
                self.buy(sl=sl, tp=tp)
                self.trades_since_cross += 1

        # Short Entry Logic
        elif self.last_cross_type == "death":
            is_trade_signal = self.vumanchu_sell[-1] == 1 and self.wt1[-1] > 0
            price_below_slow_ema = price < self.ema_slow[-1]

            if is_trade_signal and not self.htf_uptrend[-1] and volume_ok and price_below_slow_ema:
                sl = price + self.atr[-1] * self.atr_sl_multiplier
                tp = price - self.atr[-1] * self.atr_tp_multiplier
                self.sell(sl=sl, tp=tp)
                self.trades_since_cross += 1

def sanitize_stats(stats):
    """
    Sanitizes the stats object from a backtest run to make it JSON serializable.
    Removes non-serializable types like DataFrame, Timestamps, etc.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta, pd.DataFrame, type(pd.NA))):
            continue
        if isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Sanitize column names (strip whitespace and capitalize)
        df.columns = [col.strip().capitalize() for col in df.columns]

        # Preprocess the data with default parameters
        params = {
            'ema_fast_len': 50,
            'ema_slow_len': 200,
            'atr_len': 14,
            'volume_ma_len': 20
        }
        processed_df = preprocess_data(df.copy(), **params)

        # Initialize and run the backtest
        bt = Backtest(processed_df, VuManchuEMAScalping, cash=100000, commission=.002)

        print("Running backtest...")
        stats = bt.run()
        print("\nBacktest Results:")
        print(stats)

        # Save results and plot
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        # Save stats to JSON
        json_path = os.path.join(results_dir, 'temp_result.json')
        import json

        # Sanitize stats before saving
        cleaned_stats = sanitize_stats(stats)
        if '_strategy' in cleaned_stats:
            del cleaned_stats['_strategy'] # remove strategy object

        with open(json_path, 'w') as f:
            json.dump(cleaned_stats, f, indent=4)
        print(f"\nSaved strategy statistics to {json_path}")

        # Save plot
        plot_filename = os.path.join(results_dir, 'strategy_58534f4bd4d6.html')
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Saved plot to {plot_filename}")
