"""
Strategy: Market Cipher B Large Timeframe Long Continuation
"""
from backtesting import Strategy, Backtest
import pandas as pd
import pandas_ta as ta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame, **params) -> pd.DataFrame:
    """Adds indicators to the input DataFrame."""
    # Add VuManchu Cipher B indicators
    df = cipher_b(df)

    # Add other indicators required by guidelines
    df.ta.atr(length=14, append=True)
    df['volume_ma'] = df['Volume'].rolling(20).mean()

    # Higher timeframe trend filter (4H EMA 200)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['ema_200'] = ta.ema(df_4h['Close'], length=200)
    df_4h['htf_uptrend'] = df_4h['Close'] > df_4h['ema_200']

    # Merge HTF trend back to the main timeframe
    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill')

    return df

class MarketCipherBLargeTimeframeLongContinuation(Strategy):
    """
    Implements the Market Cipher B long continuation strategy for large timeframes.
    """
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    wt_threshold = 60

    def init(self):
        """Initialize indicators."""
        # Access data series directly, not through self.data.df
        self.wt1 = self.I(lambda x: x, self.data.wt1, name='wt1')
        self.atr = self.I(lambda x: x, self.data.ATRr_14, name='atr')
        self.volume_ma = self.I(lambda x: x, self.data.volume_ma, name='volume_ma')
        self.htf_uptrend = self.I(lambda x: x, self.data.htf_uptrend, name='htf_uptrend')

    def next(self):
        """Define the trading logic."""
        price = self.data.Close[-1]

        # Trend and volume filters
        if not self.htf_uptrend[-1]:
            return
        if self.data.Volume[-1] < self.volume_ma[-1]:
            return

        # Entry condition
        if not self.position and self.wt1[-1] > self.wt_threshold:
            sl = price - self.atr[-1] * self.atr_sl_multiplier
            tp = price + self.atr[-1] * self.atr_tp_multiplier
            self.buy(sl=sl, tp=tp)

        # Exit condition
        if self.position.is_long and self.wt1[-1] < self.wt_threshold:
            self.position.close()

if __name__ == '__main__':
    import json
    import os

    # Configuration
    data_path = 'data/BTC-USD-15m.csv'
    output_plot_path = 'results/market_cipher_b_large_timeframe_long_continuation.html'
    output_json_path = 'results/temp_result.json'

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Load data
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Preprocess data
    df = preprocess_data(df)

    # Run backtest
    bt = Backtest(df, MarketCipherBLargeTimeframeLongContinuation, cash=100_000, commission=.002)
    stats = bt.run()

    print(stats)

    # Save plot
    bt.plot(filename=output_plot_path, open_browser=False)

    # Save stats to JSON
    # Sanitize stats for JSON serialization
    sanitized_stats = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized_stats[key] = str(value)
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            continue # Skip non-serializable objects
        elif pd.isna(value):
            sanitized_stats[key] = None
        else:
            try:
                json.dumps(value)
                sanitized_stats[key] = value
            except (TypeError, OverflowError):
                sanitized_stats[key] = str(value)

    with open(output_json_path, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print(f"\\nResults saved to {output_json_path}")
    print(f"Plot saved to {output_plot_path}")
