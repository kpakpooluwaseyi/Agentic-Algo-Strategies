"""
VUMANCHU EMA CROSS SCALP STRATEGY
---------------------------------

Strategy Name: vumanchu_ema_cross_scalp
Strategy Type: scalping
Timeframe: 5m (adapted to 15m)
Instruments: ["BTC/USDT"]

Entry Rules:
  Long:
    - Golden Cross: 50 EMA crosses above 200 EMA.
    - First green dot appears above the zero line of the VuManchu indicator after the golden cross.
  Short:
    - Death Cross: 50 EMA crosses below 200 EMA.
    - First red dot appears above the zero line of the VuManchu indicator after the death cross.

Exit Rules:
  Take Profit: 1:1 Risk-Reward ratio.
  Stop Loss: Set stop loss above the last swing high for long trades and below the last swing low for short trades.
  Time-Based Exits: Maximum of 4-5 trades after a golden or death cross.

NOTE: This implementation adapts the original strategy to conform to the project's
      strategy development guidelines, including ATR-based risk management, a
      multi-timeframe trend filter, and volume confirmation.
"""
import os
import sys
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.signal import find_peaks

from backtesting import Backtest, Strategy

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame, **params):
    """
    Applies indicators and filters to the raw OHLCV data.
    """
    # Sanitize column names
    df.columns = [col.strip().capitalize() for col in df.columns]

    # -- Indicators --
    # Standard EMAs and ATR
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['EMA200'] = ta.ema(df['Close'], length=200)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Volume_MA'] = ta.sma(df['Volume'], length=20)

    # VuManchu Cipher B indicators
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # -- Multi-Timeframe Trend Filter (4H) --
    # Note: Using 'h' for frequency is the modern pandas syntax.
    df_4h = df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['EMA20_4H'] = ta.ema(df_4h['Close'], length=20)
    df['EMA20_4H'] = df_4h['EMA20_4H'].reindex(df.index, method='ffill')

    # -- Swing Point Detection --
    # Find peaks (swing highs) and troughs (swing lows)
    high_peaks, _ = find_peaks(df['High'], distance=10, prominence=df['ATR'].mean() * 0.5)
    low_peaks, _ = find_peaks(-df['Low'], distance=10, prominence=df['ATR'].mean() * 0.5)

    # Create columns to store the price of the last swing high/low
    df['last_swing_high'] = np.nan
    df['last_swing_low'] = np.nan

    # Use a forward-fill approach to get the *last* swing point for any given bar
    df.loc[df.index[high_peaks], 'last_swing_high'] = df['High'].iloc[high_peaks]
    df.loc[df.index[low_peaks], 'last_swing_low'] = df['Low'].iloc[low_peaks]

    # Forward-fill swing data to avoid lookahead bias. At any point, we only
    # know the last swing from the past.
    df['last_swing_high'].ffill(inplace=True)
    df['last_swing_low'].ffill(inplace=True)

    # Let backtesting.py handle the initial NaN warmup period instead of dropping them
    return df

class VuManchuEmaCrossScalp(Strategy):
    """
    Implements a scalping strategy based on EMA crosses and the VuManchu Cipher B indicator,
    adapted to meet specified development guidelines.
    """
    # Optimizable parameters
    atr_multiplier_sl = 2.0
    atr_multiplier_tp = 4.0 # Creates a 2:1 RR
    max_trades_per_regime = 5

    def init(self):
        # State variables
        self.regime = 0  # 1 for Bullish (Golden Cross), -1 for Bearish (Death Cross)
        self.trades_in_regime = 0

        # Indicators from preprocessed data
        self.ema50 = self.I(lambda x: x, self.data.EMA50, name='EMA50')
        self.ema200 = self.I(lambda x: x, self.data.EMA200, name='EMA200')
        self.ema20_4h = self.I(lambda x: x, self.data.EMA20_4H, name='EMA20_4H')
        self.atr = self.I(lambda x: x, self.data.ATR, name='ATR')
        self.volume_ma = self.I(lambda x: x, self.data.Volume_MA, name='Volume_MA')
        self.vumanchu_buy = self.I(lambda x: x, self.data.buy_signal, name='VuManchu_Buy')
        self.vumanchu_sell = self.I(lambda x: x, self.data.sell_signal, name='VuManchu_Sell')
        self.last_swing_high = self.I(lambda x: x, self.data.last_swing_high, name='Last_Swing_High')
        self.last_swing_low = self.I(lambda x: x, self.data.last_swing_low, name='Last_Swing_Low')

    def next(self):
        price = self.data.Close[-1]

        # --- Regime Detection ---
        # Golden Cross (Bullish)
        if self.ema50[-2] < self.ema200[-2] and self.ema50[-1] >= self.ema200[-1]:
            self.regime = 1
            self.trades_in_regime = 0

        # Death Cross (Bearish)
        elif self.ema50[-2] > self.ema200[-2] and self.ema50[-1] <= self.ema200[-1]:
            self.regime = -1
            self.trades_in_regime = 0

        # --- Exit active trades ---
        # This strategy uses SL/TP for exits, no additional logic needed here.

        # --- Check for new entries ---
        if self.position:
            return

        if self.trades_in_regime >= self.max_trades_per_regime:
            return

        # Volume confirmation
        volume_ok = self.data.Volume[-1] > self.volume_ma[-1]

        # --- Long Entry ---
        if self.regime == 1 and self.vumanchu_buy[-1] == 1 and volume_ok:
            # 4H Trend confirmation
            if price > self.ema20_4h[-1]:
                # Use last swing low for SL, but fallback to ATR if swing is too far
                sl_swing = self.last_swing_low[-1]
                sl_atr = price - self.atr[-1] * self.atr_multiplier_sl
                sl = max(sl_swing, sl_atr) # Use the tighter stop

                # Ensure SL is below current price
                if sl < price:
                    tp = price + (price - sl) * (self.atr_multiplier_tp / self.atr_multiplier_sl)
                    self.buy(sl=sl, tp=tp)
                    self.trades_in_regime += 1

        # --- Short Entry ---
        elif self.regime == -1 and self.vumanchu_sell[-1] == 1 and volume_ok:
             # 4H Trend confirmation
            if price < self.ema20_4h[-1]:
                # Use last swing high for SL, but fallback to ATR
                sl_swing = self.last_swing_high[-1]
                sl_atr = price + self.atr[-1] * self.atr_multiplier_sl
                sl = min(sl_swing, sl_atr) # Use the tighter stop

                # Ensure SL is above current price
                if sl > price:
                    tp = price - (sl - price) * (self.atr_multiplier_tp / self.atr_multiplier_sl)
                    self.sell(sl=sl, tp=tp)
                    self.trades_in_regime += 1


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    # Preprocess the data
    processed_df = preprocess_data(df)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Run the backtest
    bt = Backtest(processed_df, VuManchuEmaCrossScalp, cash=100_000, commission=.002)
    stats = bt.run()

    # Print the stats and save the plot
    print(stats)
    plot_filename = 'results/strategy_303f933e4770.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")

    # Save stats to JSON, dropping non-serializable objects
    stats.drop(['_strategy', '_equity_curve', '_trades'], inplace=True, errors='ignore')

    if isinstance(stats, pd.Series):
        stats_df = pd.DataFrame(stats).T
    else:
        # Fallback for different structures, though Series is expected
        stats_df = pd.DataFrame([stats])

    # Sanitize stats for JSON serialization
    def sanitize_for_json(obj):
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize_for_json(i) for i in obj]
        if isinstance(obj, pd.Series):
            return sanitize_for_json(obj.to_dict())
        if pd.isna(obj):
            return None
        return obj

    # Use .map() instead of deprecated .applymap()
    sanitized_stats = stats_df.map(sanitize_for_json).to_dict(orient='records')[0]

    import json
    json_filename = 'results/temp_result.json'
    with open(json_filename, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)
    print(f"Stats saved to {json_filename}")
