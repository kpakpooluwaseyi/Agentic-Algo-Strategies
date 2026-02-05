"""
Strategy: Accumulation/Distribution Breakout (VuManchu Proxy)
Source: Peter Wyckoff - The Psychology of Stock Market Timing (1963)
Author: Moon Dev
"""
import pandas as pd
import numpy as np
import json
import os
import sys
from backtesting import Strategy, Backtest

# Add parent directory to path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indicators.vumanchu import cipher_b
import talib

def preprocess_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Applies the mandated VuManchu Cipher B indicator and ATR for risk management.
    """
    df = df.copy()
    # Apply the Cipher B indicator suite
    df = cipher_b(df)

    # Add ATR for risk management, as per guidelines
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=params.get('risk_atr_period', 14))

    # Convert boolean signals to int for backtesting.py compatibility
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)
    return df

class AccumulationDistributionBreakout(Strategy):
    """
    A proxy for Wyckoff's Accumulation/Distribution Breakout using VuManchu Cipher B.
    - Consolidation is proxied by WaveTrend in overbought/oversold zones.
    - Breakouts are proxied by WaveTrend crosses.
    - Volume confirmation is proxied by the RSI+MFI indicator.
    """
    # Optimizable parameters
    risk_atr_period = 14
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 4.0

    def init(self):
        """Initialize all indicators using self.I() for backtesting.py compatibility."""
        self.buy_sig = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_sig = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.mf = self.I(lambda: self.data.rsimfi, name='money_flow')
        self.atr = self.I(lambda: self.data.atr, name='atr')

    def next(self):
        """Define the trading logic for each bar."""
        # Skip warmup period for indicators
        if len(self.data) < 65: # MFI needs 60 + SMMA warmup
            return

        current_price = self.data.Close[-1]
        atr_val = self.atr[-1]

        # Entry logic
        if not self.position:
            # Long Entry Proxy: Buy signal with positive money flow
            if self.buy_sig[-1] == 1 and self.mf[-1] > 0:
                sl = current_price - (self.atr_sl_multiplier * atr_val)
                tp = current_price + (self.atr_tp_multiplier * atr_val)
                self.buy(sl=sl, tp=tp)

            # Short Entry Proxy: Sell signal with negative money flow
            elif self.sell_sig[-1] == 1 and self.mf[-1] < 0:
                sl = current_price + (self.atr_sl_multiplier * atr_val)
                tp = current_price - (self.atr_tp_multiplier * atr_val)
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    plot_filename = os.path.join(results_dir, 'accumulation_distribution_breakout.html')
    json_filename = os.path.join(results_dir, 'temp_result.json')
    os.makedirs(results_dir, exist_ok=True)

    try:
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        df.columns = [col.strip().capitalize() for col in df.columns]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    except FileNotFoundError:
        print("Generating synthetic data...")
        date_rng = pd.date_range(start='2022-01-01', end='2023-01-01', freq='15min')
        df = pd.DataFrame(index=date_rng)
        price = 100 + np.random.randn(len(df)).cumsum()
        df['Open'] = price
        df['High'] = price + abs(np.random.randn(len(df)))
        df['Low'] = price - abs(np.random.randn(len(df)))
        df['Close'] = price + np.random.randn(len(df))
        df['Volume'] = 100 + np.random.rand(len(df)) * 100

    strategy_params = {'risk_atr_period': AccumulationDistributionBreakout.risk_atr_period}
    df = preprocess_data(df, strategy_params)
    df.dropna(inplace=True)

    bt = Backtest(df, AccumulationDistributionBreakout, cash=100_000, commission=.002)
    stats = bt.run()

    print("\n--- Backtest Results ---")
    print(stats)

    # Sanitize and save results to JSON
    stats_dict = {k: v for k, v in stats.items() if k not in ['_strategy', '_equity_curve', '_trades']}
    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)

    with open(json_filename, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"\nResults saved to {json_filename}")

    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"\nCould not generate plot. Error: {e}")
