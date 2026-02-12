"""
Strategy: fibonacci_ideal_setup
"""

import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

# --- Strategy Development Rules ---
# 1. ATR-Based Risk Management (2x SL, 3x TP)
# 2. Multi-Timeframe Trend Filter (4H EMA200)
# 3. Volume Confirmation (Volume > 20-period SMA)
# 4. No hard-coded values

def preprocess_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Adds indicators and mandatory features to the DataFrame.
    """
    # Mandatory Indicators
    # 1. Higher Timeframe Trend
    df_4h = df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['ema_200'] = ta.ema(df_4h['Close'], length=200)

    # Define trend based on price relative to EMA
    df_4h['htf_trend'] = 0
    df_4h.loc[df_4h['Close'] > df_4h['ema_200'], 'htf_trend'] = 1
    df_4h.loc[df_4h['Close'] < df_4h['ema_200'], 'htf_trend'] = -1

    # Map 4H trend to original timeframe
    df['htf_trend'] = df_4h['htf_trend'].reindex(df.index, method='ffill')
    df['htf_trend'] = df['htf_trend'].fillna(0) # Fill initial NaNs

    # 2. Volume Confirmation
    df['volume_ma'] = ta.sma(df['Volume'], length=20)

    # 3. ATR for Risk Management
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Strategy-Specific Indicators
    df['ema_34'] = ta.ema(df['Close'], length=34)
    df['cci_14'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)
    df['cci_50'] = ta.cci(df['High'], df['Low'], df['Close'], length=50)

    # Let the strategy handle initial NaNs
    return df

class FibonacciIdealSetup(Strategy):
    """
    Implements the Fibonacci Ideal Setup strategy.
    """
    # Optimizable Parameters
    atr_sl_multiplier = 2.0
    swing_lookback = 50 # For find_peaks distance
    swing_distance = 5 # For find_peaks distance

    def init(self):
        # Mandatory Indicators
        self.htf_trend = self.I(lambda: self.data.htf_trend, name="htf_trend")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="volume_ma")
        self.atr = self.I(lambda: self.data.atr, name="atr")

        # Strategy Indicators
        self.ema_34 = self.I(lambda: self.data.ema_34, name="ema_34")
        self.cci_14 = self.I(lambda: self.data.cci_14, name="cci_14")
        self.cci_50 = self.I(lambda: self.data.cci_50, name="cci_50")

    def next(self):
        # --- Data Integrity Check ---
        # Wait for all indicators to have valid values
        if (pd.isna(self.atr[-1]) or
            pd.isna(self.volume_ma[-1]) or
            pd.isna(self.ema_34[-1]) or
            pd.isna(self.cci_14[-1]) or
            pd.isna(self.cci_50[-1])):
            return

        # Ensure enough data for swing lookbacks
        if len(self.data.Close) < self.swing_lookback + 5:
            return

        # Alias current price
        price = self.data.Close[-1]

        # --- Mandatory Filter Checks ---
        # 1. Volume Confirmation
        is_volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]

        # 2. Higher Timeframe Trend Confirmation
        is_htf_uptrend = self.htf_trend[-1] == 1
        is_htf_downtrend = self.htf_trend[-1] == -1

        # --- Find Swings for Fibonacci Symmetry ---
        recent_data = self.data.df.iloc[-self.swing_lookback:]
        highs, _ = find_peaks(recent_data['High'], distance=self.swing_distance, prominence=recent_data['atr'].mean() * 0.5)
        lows, _ = find_peaks(-recent_data['Low'], distance=self.swing_distance, prominence=recent_data['atr'].mean() * 0.5)

        # --- Entry Logic ---
        if not self.position:
            # --- Long Setup ---
            if is_htf_uptrend and len(lows) >= 1:
                # C-point is the most recent swing low
                c_low_idx = lows[-1]
                c_low_price = recent_data['Low'].iloc[c_low_idx]

                # Check if current price is testing the C-point (pullback zone)
                is_testing_support = abs(price - c_low_price) / price < 0.005 # 0.5% tolerance

                if is_testing_support:
                    # Confluence Checks for Long
                    is_above_ema = price > self.ema_34[-1]
                    is_cci_positive = self.cci_14[-1] > 0 and self.cci_50[-1] > 0

                    if is_above_ema and is_cci_positive and is_volume_confirmed:
                        # Find A and B points to calculate the D target
                        if len(highs) >= 1 and any(highs < c_low_idx):
                           b_high_idx = highs[highs < c_low_idx].max()
                           if any(lows < b_high_idx):
                                a_low_idx = lows[lows < b_high_idx].max()
                                b_high_price = recent_data['High'].iloc[b_high_idx]
                                a_low_price = recent_data['Low'].iloc[a_low_idx]

                                # D target = C + (B - A)
                                tp_price = c_low_price + (b_high_price - a_low_price)
                                sl = price - self.atr[-1] * self.atr_sl_multiplier

                                # Ensure TP is higher than entry
                                if tp_price > price:
                                    self.buy(sl=sl, tp=tp_price)

            # --- Short Setup ---
            if is_htf_downtrend and len(highs) >= 1:
                # C-point is the most recent swing high
                c_high_idx = highs[-1]
                c_high_price = recent_data['High'].iloc[c_high_idx]

                # Check if current price is testing the C-point (pullback zone)
                is_testing_resistance = abs(price - c_high_price) / price < 0.005 # 0.5% tolerance

                if is_testing_resistance:
                    # Confluence Checks for Short
                    is_below_ema = price < self.ema_34[-1]
                    is_cci_negative = self.cci_14[-1] < 0 and self.cci_50[-1] < 0

                    if is_below_ema and is_cci_negative and is_volume_confirmed:
                        # Find A and B points to calculate the D target
                        if len(lows) >= 1 and any(lows < c_high_idx):
                            b_low_idx = lows[lows < c_high_idx].max()
                            if any(highs < b_low_idx):
                                a_high_idx = highs[highs < b_low_idx].max()
                                b_low_price = recent_data['Low'].iloc[b_low_idx]
                                a_high_price = recent_data['High'].iloc[a_high_idx]

                                # D target = C - (A - B)
                                tp_price = c_high_price - (a_high_price - b_low_price)
                                sl = price + self.atr[-1] * self.atr_sl_multiplier

                                # Ensure TP is lower than entry
                                if tp_price < price:
                                    self.sell(sl=sl, tp=tp_price)

def run_backtest(df: pd.DataFrame):
    """
    Runs the backtest and saves results.
    """
    bt = Backtest(df, FibonacciIdealSetup, cash=100_000, commission=.002)
    stats = bt.run()

    print("--- Backtest Results ---")
    print(stats)

    # Save results to JSON
    import json
    results_path = "results/temp_result.json"

    # Basic serialization
    serializable_stats = {k: str(v) for k, v in stats.items() if not isinstance(v, (pd.DataFrame, pd.Series))}

    with open(results_path, 'w') as f:
        json.dump(serializable_stats, f, indent=4)

    print(f"\\nResults saved to {results_path}")

    # Save plot
    plot_path = "results/fibonacci_ideal_setup.html"
    bt.plot(filename=plot_path, open_browser=False)
    print(f"Plot saved to {plot_path}")

if __name__ == '__main__':
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv',
                           parse_dates=['datetime'],
                           index_col='datetime')

        # Robustly clean and capitalize column names
        data.columns = [col.strip().capitalize() for col in data.columns]

        print("Data loaded successfully.")

        # Preprocess data
        data_processed = preprocess_data(data.copy(), {})

        if data_processed.empty:
            raise ValueError("Preprocessing resulted in an empty DataFrame. Check indicator periods and data length.")

        print("Data preprocessed successfully.")

        # Run backtest
        run_backtest(data_processed)

    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
