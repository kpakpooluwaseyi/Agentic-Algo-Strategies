
import pandas as pd
from backtesting import Strategy, Backtest
import talib
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b


def preprocess_data(df, **params):
    """
    Adds all the required indicators and filters to the DataFrame.
    """
    # Add VuManchu Cipher B indicators
    df = cipher_b(df)

    # Convert boolean signals to int for backtesting.py compatibility
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # ATR for risk management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Higher timeframe trend filter (4H)
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # 4H EMA200
    if not df_4h.empty and len(df_4h) > 200:
        df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)

        # Determine 4H trend
        df_4h['htf_uptrend'] = (df_4h['Close'] > df_4h['ema_200']).astype(int)

        # Reindex to match the original dataframe and forward fill
        df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill')
    else:
        df['htf_uptrend'] = 1 # Default to uptrend if not enough data

    # Volume Confirmation
    df['volume_ma'] = df['Volume'].rolling(20).mean()

    return df


class MeanReversionVumanchu(Strategy):
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_multiplier = 1.0

    def init(self):
        # Pre-calculated indicators
        self.wt_buy_signal = self.I(lambda: self.data.buy_signal, name='wt_buy_signal')
        self.wt_sell_signal = self.I(lambda: self.data.sell_signal, name='wt_sell_signal')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend, name='htf_uptrend')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')

    def next(self):
        # Skip warmup period and bars with missing data
        if len(self.data) < 200 or pd.isna(self.atr[-1]) or pd.isna(self.htf_uptrend[-1]) or pd.isna(self.volume_ma[-1]):
            return

        current_price = self.data.Close[-1]
        atr_value = self.atr[-1]

        # Check filters first
        is_high_volume = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_ma_multiplier

        # Entry logic
        if not self.position:
            # Long entry: Oversold signal + Uptrend + Volume confirmation
            if self.wt_buy_signal[-1] and self.htf_uptrend[-1] == 1 and is_high_volume:
                sl = current_price - (self.atr_sl_multiplier * atr_value)
                tp = current_price + (self.atr_tp_multiplier * atr_value)
                self.buy(sl=sl, tp=tp)

            # Short entry: Overbought signal + Downtrend + Volume confirmation
            elif self.wt_sell_signal[-1] and self.htf_uptrend[-1] == 0 and is_high_volume:
                sl = current_price + (self.atr_sl_multiplier * atr_value)
                tp = current_price - (self.atr_tp_multiplier * atr_value)
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    # Define the data path
    data_path = 'data/BTC-USD-15m.csv'

    # Load data
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Sanitize column names (e.g., 'open' -> 'Open')
        df.columns = [col.strip().capitalize() for col in df.columns]
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # As a fallback, create some synthetic data for demonstration
        print("Generating synthetic data...")
        from backtesting.test import EURUSD
        df = EURUSD.copy()
        df = df.iloc[-5000:] # Use a subset for speed

    # Preprocess the data
    df = preprocess_data(df)

    # Clean data from NaNs
    df.dropna(inplace=True)

    # Ensure dataframe is not empty after preprocessing
    if df.empty:
        print("Error: DataFrame is empty after preprocessing. Cannot run backtest.")
    else:
        # Run the backtest
        bt = Backtest(df, MeanReversionVumanchu, cash=100_000, commission=.002)
        stats = bt.run()

        print("\n" + "="*80)
        print("Mean Reversion Vumanchu Strategy Results")
        print("="*80)
        print(stats)

        # Save results to JSON
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Sanitize stats for JSON serialization
        stats_serializable = {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in stats.items() if not isinstance(v, (pd.Series, pd.DataFrame))}

        with open(os.path.join(results_dir, 'temp_result.json'), 'w') as f:
            import json
            json.dump(stats_serializable, f, indent=4)

        print(f"\nResults saved to {os.path.join(results_dir, 'temp_result.json')}")

        # Save the plot
        plot_filename = os.path.join(results_dir, 'mean_reversion_vumanchu.html')
        try:
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not save plot: {e}")
