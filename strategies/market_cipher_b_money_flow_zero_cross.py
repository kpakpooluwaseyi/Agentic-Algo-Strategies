"""
Strategy: Market Cipher B Money Flow Zero Cross
Author: Jules
Date: 2024-07-25

Description:
This strategy enters trades based on the Money Flow Index (MFI) from the Market Cipher B
indicator suite crossing the zero line. It uses confirmation from the WaveTrend oscillator's
buy/sell dots, a higher-timeframe trend filter, and volume confirmation. Risk management
is handled by ATR-based stop-loss and take-profit levels.

Entry Rules:
Long:
1. Money Flow (rsimfi) crosses above the zero line.
2. A green dot (buy_signal) appears as confirmation.
3. Price is above the 4-hour 200 SMA (trend filter).
4. Volume is above its 20-period SMA (volume confirmation).

Short:
1. Money Flow (rsimfi) crosses below the zero line.
2. A red dot (sell_signal) appears as confirmation.
3. Price is below the 4-hour 200 SMA (trend filter).
4. Volume is above its 20-period SMA (volume confirmation).

Exit Rules:
- Stop Loss: 2 * ATR from the entry price.
- Take Profit: 3 * ATR from the entry price.
- Early Exit: Position is closed if an opposite signal occurs (e.g., money flow crosses zero
  in the opposite direction or an opposite dot appears).
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Add src directory to path to import vumanchu
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame, **params):
    """
    Applies the necessary indicators to the input DataFrame.
    - Market Cipher B suite
    - ATR for risk management
    - Higher timeframe SMA for trend filtering
    - Volume SMA for confirmation
    """
    df = df.copy()

    # 1. Apply Market Cipher B indicators
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # 2. Calculate ATR
    df.ta.atr(length=params.get('atr_period', 14), append=True)

    # 3. Calculate higher timeframe (4H) trend filter
    # Resample to 4H, calculate SMA, and then merge back to 15m
    sma_period = params.get('sma_period', 200)
    df_4h = df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h[f'SMA_{sma_period}_4h'] = df_4h['Close'].rolling(window=sma_period).mean()
    print(f"df_4h shape: {df_4h.shape}")
    print(f"Non-NaN SMA values in df_4h: {df_4h[f'SMA_{sma_period}_4h'].notna().sum()}")

    # Forward fill the 4h SMA to the 15m timeframe
    df[f'SMA_{sma_period}_4h'] = df_4h[f'SMA_{sma_period}_4h'].reindex(df.index, method='ffill')
    print(f"Shape after merging 4h SMA: {df.shape}")
    print(f"Non-NaN SMA values in merged df: {df[f'SMA_{sma_period}_4h'].notna().sum()}")

    # 4. Calculate Volume SMA for confirmation
    df['Volume_SMA'] = df['Volume'].rolling(window=params.get('volume_sma_period', 20)).mean()

    # Let backtesting.py handle NaN warmup periods
    # df.dropna(inplace=True)
    return df


class MarketCipherBMoneyFlowZeroCross(Strategy):
    """
    Implements the Market Cipher B Money Flow Zero Cross strategy.
    """
    # Optimizable parameters
    atr_period = 14
    sma_period = 200
    volume_sma_period = 20
    sl_atr_multiplier = 2.0
    tp_atr_multiplier = 3.0

    def init(self):
        # Initialize indicators using self.I()
        self.mf = self.I(lambda: self.data.rsimfi, name="MoneyFlow")
        self.buy_dot = self.I(lambda: self.data.buy_signal, name="BuyDot")
        self.sell_dot = self.I(lambda: self.data.sell_signal, name="SellDot")

        self.atr = self.I(lambda: self.data[f'ATRr_{self.atr_period}'], name="ATR")
        self.sma_4h = self.I(lambda: self.data[f'SMA_{self.sma_period}_4h'], name="SMA_4h")
        self.volume_sma = self.I(lambda: self.data.Volume_SMA, name="Volume_SMA")

    def next(self):
        # Skip if there is no trend signal yet
        if np.isnan(self.sma_4h[-1]):
            return

        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        atr_val = self.atr[-1]

        # --- ENTRY CONDITIONS ---
        is_uptrend = price > self.sma_4h[-1]
        is_downtrend = price < self.sma_4h[-1]
        has_volume = volume > self.volume_sma[-1]

        # Long Entry: Money Flow crosses above zero, with confirmations
        if not self.position and crossover(self.mf, 0) and self.buy_dot[-1] == 1 and is_uptrend and has_volume:
            sl = price - self.sl_atr_multiplier * atr_val
            tp = price + self.tp_atr_multiplier * atr_val
            self.buy(sl=sl, tp=tp)

        # Short Entry: Money Flow crosses below zero, with confirmations
        elif not self.position and crossover(0, self.mf) and self.sell_dot[-1] == 1 and is_downtrend and has_volume:
            sl = price + self.sl_atr_multiplier * atr_val
            tp = price - self.tp_atr_multiplier * atr_val
            self.sell(sl=sl, tp=tp)

        # --- EARLY EXIT CONDITIONS ---
        if self.position.is_long:
            # Exit long if MF crosses back below zero or a sell dot appears
            if crossover(0, self.mf) or self.sell_dot[-1] == 1:
                self.position.close()

        elif self.position.is_short:
            # Exit short if MF crosses back above zero or a buy dot appears
            if crossover(self.mf, 0) or self.buy_dot[-1] == 1:
                self.position.close()


def sanitize_stats_for_json(stats):
    """
    Converts a backtesting.py stats Series to a JSON-serializable dictionary,
    skipping complex objects like DataFrames.
    """
    sanitized = {}
    for key, value in stats.items():
        # Skip DataFrames and other complex objects
        if isinstance(value, (pd.DataFrame, pd.Series, Strategy)):
            continue

        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = value.item()
        elif pd.isna(value):
            sanitized[key] = None
        elif isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        else:
            # Fallback for other potential non-serializable types
            try:
                json.dumps(value)
                sanitized[key] = value
            except TypeError:
                continue
    return sanitized

# --- Backtesting Runner ---
if __name__ == '__main__':
    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Load data
    data_path = Path("data/BTC-USD-15m.csv")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    # Sanitize column names (e.g., '  open  ' -> 'Open')
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Preprocess data with default parameters
    params = {
        'atr_period': 14,
        'sma_period': 200,
        'volume_sma_period': 20
    }
    processed_df = preprocess_data(df, **params)

    if processed_df.empty:
        print("Error: Preprocessing resulted in an empty DataFrame. Check data and indicator periods.")
        sys.exit(1)

    # Run backtest
    bt = Backtest(
        processed_df,
        MarketCipherBMoneyFlowZeroCross,
        cash=100_000,
        commission=.002,
        trade_on_close=True,
        finalize_trades=True
    )

    stats = bt.run()

    print("\n--- Backtest Results ---")
    print(stats)

    # Save stats to JSON
    stats_dict = sanitize_stats_for_json(stats)

    json_path = results_dir / "temp_result.json"
    with open(json_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"\nStats saved to {json_path}")

    # Generate and save plot
    plot_path = results_dir / "market_cipher_b_money_flow_zero_cross.html"
    bt.plot(filename=str(plot_path), open_browser=False)
    print(f"Plot saved to {plot_path}")
