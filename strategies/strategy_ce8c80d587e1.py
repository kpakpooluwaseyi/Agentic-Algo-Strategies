"""
Strategy: support_resistance_reversal_macd_stoch_rsi_short
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Strategy, Backtest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df, **params):
    """
    Adds indicators and higher-timeframe data to the dataframe.
    """
    # Add Cipher B indicators
    df = cipher_b(df)

    # Add MACD
    df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)

    # ATR for risk management
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)


    # Higher-timeframe trend filter (4h)
    df_4h = df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = ta.ema(df_4h['Close'], length=200)
    df_4h['htf_trend_down'] = (df_4h['Close'] < df_4h['ema_200']).astype(int)

    # Merge HTF trend back to the original dataframe
    df = pd.merge(df, df_4h[['htf_trend_down']], how='left', left_index=True, right_index=True)
    df['htf_trend_down'] = df['htf_trend_down'].ffill()

    # For backtesting.py, boolean signals should be converted to int
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    return df

class SupportResistanceReversal(Strategy):
    """
    Implements the support resistance reversal strategy for short entries.
    """
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    stoch_rsi_overbought = 80
    resistance_lookback = 20

    def init(self):
        """
        Initialize indicators and state variables.
        """
        self.htf_trend_down = self.I(lambda: self.data.htf_trend_down, name='htf_trend_down')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.stoch_rsi_k = self.I(lambda: self.data.stoch_rsi_k, name='stoch_rsi_k')
        self.stoch_rsi_d = self.I(lambda: self.data.stoch_rsi_d, name='stoch_rsi_d')
        self.macd = self.I(lambda: self.data['MACD_12_26_9'], name='macd')
        self.macd_signal = self.I(lambda: self.data['MACDs_12_26_9'], name='macd_signal')
        self.entry_cooldown = 0

    def next(self):
        """
        Define entry and exit logic.
        """
        price = self.data.Close[-1]

        # Cooldown logic for delayed entry
        if self.entry_cooldown > 0:
            self.entry_cooldown -= 1
            if self.entry_cooldown == 0 and not self.position:
                sl = price + self.atr[-1] * self.atr_sl_multiplier
                tp = price - self.atr[-1] * self.atr_tp_multiplier
                self.sell(sl=sl, tp=tp)
            return

        # Bearish engulfing pattern
        is_bullish_prev = self.data.Close[-2] > self.data.Open[-2]
        is_bearish_curr = self.data.Close[-1] < self.data.Open[-1]
        engulfs = self.data.Open[-1] > self.data.Close[-2] and self.data.Close[-1] < self.data.Open[-2]
        bearish_engulfing = is_bullish_prev and is_bearish_curr and engulfs

        # Resistance zone check
        highest_high = self.data.High[-self.resistance_lookback:].max()
        is_at_resistance = self.data.High[-1] >= highest_high

        # Entry setup conditions
        if (not self.position and
            self.entry_cooldown == 0 and
            self.htf_trend_down[-1] == 1 and
            is_at_resistance and
            bearish_engulfing and
            self.stoch_rsi_k[-1] > self.stoch_rsi_overbought and
            self.stoch_rsi_d[-1] > self.stoch_rsi_overbought and
            self.stoch_rsi_k[-2] > self.stoch_rsi_d[-2] and self.stoch_rsi_k[-1] < self.stoch_rsi_d[-1] and # Bearish cross
            self.data.Volume[-1] < self.data.Volume[-2] and # Volume is decreasing
            (self.macd[-1] > self.macd_signal[-1]) and # MACD is still above signal
            ((self.macd[-1] - self.macd_signal[-1]) < (self.macd[-2] - self.macd_signal[-2]))): # Gap is narrowing

            # Activate cooldown for entry on the second candle after this one
            self.entry_cooldown = 2

if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        # Sanitize column names
        df.columns = [col.strip().capitalize() for col in df.columns]
        if 'Unnamed: 6' in df.columns:
            df.drop(columns=['Unnamed: 6'], inplace=True)
    except FileNotFoundError:
        print("Data file not found. A sample dataset will be generated.")
        # Generate sample data if the file doesn't exist
        dates = pd.date_range('2023-01-01', periods=2000, freq='15min')
        np.random.seed(42)
        price = 20000 + np.cumsum(np.random.randn(2000) * 10)
        df = pd.DataFrame({
            'open': price,
            'high': price + np.random.rand(2000) * 20,
            'low': price - np.random.rand(2000) * 20,
            'close': price + np.random.randn(2000) * 5,
            'volume': np.random.rand(2000) * 1000
        }, index=dates)
        # Rename columns to match backtesting.py expectations
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)


    # Preprocess data
    df_processed = preprocess_data(df.copy())

    # Drop NaN rows
    df_processed.dropna(inplace=True)

    # Run backtest
    bt = Backtest(df_processed, SupportResistanceReversal, cash=100_000, commission=.002)
    stats = bt.run()

    print(stats)

    # Save results to a JSON file
    import json
    results_path = 'results/temp_result.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Sanitize stats object for JSON serialization
    sanitized_stats = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized_stats[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized_stats[key] = float(value)
        elif isinstance(value, pd.DataFrame):
            # Skip DataFrames like _equity_curve and _trades
            continue
        elif isinstance(value, Strategy):
            # Convert strategy object to its string representation
            sanitized_stats[key] = str(value)
        else:
            sanitized_stats[key] = value

    with open(results_path, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print(f"Results saved to {results_path}")

    # Generate and save the plot
    plot_path = 'results/strategy_ce8c80d587e1.html'
    bt.plot(filename=plot_path, open_browser=False)
    print(f"Plot saved to {plot_path}")
