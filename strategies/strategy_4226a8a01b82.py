import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest
import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df, **params):
    """
    Adds all required indicators and filters to the DataFrame.
    """
    df = df.copy()

    # Apply VuManchu Cipher B indicators
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # Higher Timeframe Trend (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    # Ensure the resampled DataFrame is not empty
    if not df_4h.empty:
        df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
        df_4h['htf_trend'] = (df_4h['Close'] > df_4h['ema_200']).astype(int)

        # Merge HTF trend back into the original DataFrame
        df = df.merge(df_4h[['htf_trend']], left_index=True, right_index=True, how='left')
        df['htf_trend'] = df['htf_trend'].ffill()
        # Fill initial NaNs that ffill might miss
        df['htf_trend'] = df['htf_trend'].bfill()
    else:
        # If not enough data for 4H resampling, create a placeholder column
        df['htf_trend'] = 0

    # Volume Confirmation
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # ATR for Risk Management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    return df

class QuantitativeMomentumStrategy(Strategy):
    """
    A momentum strategy based on VuManchu Cipher B signals, confirmed by
    higher timeframe trend and volume.
    """
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_multiplier = 1.0

    def init(self):
        self.buy_sig = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_sig = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.htf_trend = self.I(lambda: self.data.htf_trend, name='htf_trend')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
        self.atr = self.I(lambda: self.data.atr, name='atr')

    def next(self):
        # Skip if any indicator data is not ready
        if np.isnan(self.htf_trend[-1]) or np.isnan(self.volume_ma[-1]) or np.isnan(self.atr[-1]):
            return

        price = self.data.Close[-1]

        # --- Entry Conditions ---
        if not self.position:
            # Long Entry
            is_buy_signal = self.buy_sig[-1] == 1
            is_uptrend = self.htf_trend[-1] == 1
            is_high_volume = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_ma_multiplier

            if is_buy_signal and is_uptrend and is_high_volume:
                sl = price - (self.atr_sl_multiplier * self.atr[-1])
                tp = price + (self.atr_tp_multiplier * self.atr[-1])
                self.buy(sl=sl, tp=tp)

            # Short Entry
            is_sell_signal = self.sell_sig[-1] == 1
            is_downtrend = self.htf_trend[-1] == 0

            if is_sell_signal and is_downtrend and is_high_volume:
                sl = price + (self.atr_sl_multiplier * self.atr[-1])
                tp = price - (self.atr_tp_multiplier * self.atr[-1])
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    # --- Backtesting Setup ---
    DATA_PATH = 'data/BTC-USD-15m.csv'

    try:
        df = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)

    # --- Column Name Sanitization ---
    # The vumanchu library expects title-cased column names.
    df.columns = [col.strip().title() for col in df.columns]

    # Remove the unnamed column if it exists
    if 'Unnamed: 6' in df.columns:
        df = df.drop(columns=['Unnamed: 6'])

    # --- Preprocessing ---
    df = preprocess_data(df)
    df = df.dropna()

    # --- Run Backtest ---
    bt = Backtest(df, QuantitativeMomentumStrategy, cash=100_000, commission=.002)
    stats = bt.run()

    print("\n--- Backtest Results ---")
    print(stats)

    # --- Save Results ---
    def sanitize_stats(stats):
        """Converts a backtesting.py stats Series to a JSON-serializable dict."""
        stats_dict = stats.to_dict()

        # Remove non-serializable objects
        stats_dict.pop('_strategy', None)
        stats_dict.pop('_equity_curve', None)
        stats_dict.pop('_trades', None)

        for key, value in stats_dict.items():
            if isinstance(value, pd.Timestamp):
                stats_dict[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                stats_dict[key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                stats_dict[key] = float(value)
            elif pd.isna(value):
                stats_dict[key] = None
        return stats_dict

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Sanitize and save stats to JSON
    sanitized_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print("\nBacktest stats saved to results/temp_result.json")

    # Save plot
    plot_filename = 'results/strategy_4226a8a01b82.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Backtest plot saved to {plot_filename}")
