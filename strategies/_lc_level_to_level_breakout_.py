"""
Level-to-Level Breakout Strategy
================================

This strategy identifies consolidation zones (supply and demand levels) and trades
breakouts, confirming the move with multi-timeframe trend and momentum analysis
using the VuManchu Cipher B "Cloud".

Strategy Logic:
- Predefines supply and demand zones based on recent price action.
- Waits for a clean breakout above supply or below demand.
- Confirms the breakout with VuManchu Cloud alignment on multiple timeframes.
- Enforces ATR-based risk management and volume confirmation.
"""

from backtesting import Strategy
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b
import pandas_ta as ta


def preprocess_data(df, **params):
    """
    Adds necessary indicators for the LcLevelToLevelBreakout strategy.
    - VuManchu Cipher B for momentum and "Cloud"
    - Rolling pivots for supply/demand zones
    - Higher-timeframe SMA proxy for trend filter
    - Rolling volume average for confirmation
    - ATR for risk management
    """
    # Get parameters or use defaults from the class
    zone_lookback = params.get('zone_lookback', 20)
    atr_period = params.get('atr_period', 14)
    htf_trend_period = params.get('htf_trend_period', 200)

    df = df.copy()

    # 1. VuManchu Cipher B Cloud
    df = cipher_b(df)

    # 2. Supply/Demand Zones (Rolling Pivots)
    df['supply'] = df['High'].rolling(zone_lookback, min_periods=2).max()
    df['demand'] = df['Low'].rolling(zone_lookback, min_periods=2).min()

    # 3. Higher Timeframe Trend Filter (Proxy)
    df['sma_htf'] = df['Close'].rolling(htf_trend_period).mean()

    # 4. Volume Confirmation
    df['volume_avg'] = df['Volume'].rolling(zone_lookback).mean()

    # 5. ATR for Risk Management
    # pandas-ta appends a column named 'ATRr_14' by default
    df.ta.atr(length=atr_period, append=True)

    return df


class LcLevelToLevelBreakout(Strategy):
    """
    Implements the Level-to-Level Breakout strategy.

    Entry (Long):
    - Price breaks cleanly above a pre-defined supply level.
    - VuManchu "Cloud" (wt1 > wt2) is bullish on execution and higher TFs.
    - 4H trend is up (Close > SMA_4H).
    - Volume is above average.

    Entry (Short):
    - Price breaks cleanly below a pre-defined demand level.
    - VuManchu "Cloud" (wt1 < wt2) is bearish on execution and higher TFs.
    - 4H trend is down (Close < SMA_4H).
    - Volume is above average.
    """

    # ===== OPTIMIZABLE PARAMETERS =====
    zone_lookback = 20      # Lookback period for defining supply/demand zones
    htf_trend_period = 200  # Lookback for the higher-timeframe trend SMA
    atr_period = 14         # ATR calculation period
    sl_atr_multiplier = 2   # ATR multiplier for stop loss
    tp_atr_multiplier = 3   # ATR multiplier for take profit

    def init(self):
        """
        Initialize indicators here.
        """
        # VuManchu Cloud
        self.wt1 = self.I(lambda: self.data.wt1, name="WaveTrend1")
        self.wt2 = self.I(lambda: self.data.wt2, name="WaveTrend2")

        # Supply/Demand Zones
        self.supply = self.I(lambda: self.data.supply, name="SupplyZone")
        self.demand = self.I(lambda: self.data.demand, name="DemandZone")

        # HTF Trend and Volume
        self.sma_htf = self.I(lambda: self.data.sma_htf, name="SMA_HTF")
        self.volume_avg = self.I(lambda: self.data.volume_avg, name="VolumeAvg")

        # ATR
        # The column name from pandas_ta is based on the period, e.g., 'ATRr_14'
        atr_col_name = f'ATRr_{self.atr_period}'
        self.atr = self.I(lambda: self.data[atr_col_name], name="ATR")

    def next(self):
        """
        Main trading logic. Called on each new bar.
        """
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]

        # Define the bullish "Cloud" condition
        is_bullish_cloud = self.wt1[-1] > self.wt2[-1]

        # Define the bearish "Cloud" condition
        is_bearish_cloud = self.wt1[-1] < self.wt2[-1]

        # Trend and Volume confirmation
        is_htf_uptrend = price > self.sma_htf[-1]
        is_htf_downtrend = price < self.sma_htf[-1]
        is_volume_strong = volume > self.volume_avg[-1]

        # Avoid trading if already in a position
        if self.position:
            return

        # --- LONG ENTRY ---
        # Condition: Clean breakout above the supply zone
        if self.data.High[-1] > self.supply[-2] and self.data.Close[-1] > self.data.Open[-1]:
            # Confirmation: Bullish cloud, HTF uptrend, and strong volume
            if is_bullish_cloud and is_htf_uptrend and is_volume_strong:
                sl = price - self.atr[-1] * self.sl_atr_multiplier
                tp = price + self.atr[-1] * self.tp_atr_multiplier

                # Ensure TP and SL are valid
                if tp > price and sl < price:
                    self.buy(sl=sl, tp=tp)

        # --- SHORT ENTRY ---
        # Condition: Clean breakdown below the demand zone
        elif self.data.Low[-1] < self.demand[-2] and self.data.Close[-1] < self.data.Open[-1]:
            # Confirmation: Bearish cloud, HTF downtrend, and strong volume
            if is_bearish_cloud and is_htf_downtrend and is_volume_strong:
                sl = price + self.atr[-1] * self.sl_atr_multiplier
                tp = price - self.atr[-1] * self.tp_atr_multiplier

                # Ensure TP and SL are valid
                if tp < price and sl > price:
                    self.sell(sl=sl, tp=tp)


# ===== STANDALONE MODE =====
if __name__ == '__main__':
    from backtesting import Backtest
    import pandas as pd

    # Load sample data
    try:
        # The user requested 'data/BTC-USD-15m.csv'
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("No data file found at 'data/BTC-USD-15m.csv'. Please check the path.")
        exit(1)

    # Preprocess if needed
    df = preprocess_data(df)

    # Run backtest
    bt = Backtest(df, LcLevelToLevelBreakout, cash=100000, commission=0.001)
    stats = bt.run()
    print(stats)

    # Show plot
    bt.plot()
