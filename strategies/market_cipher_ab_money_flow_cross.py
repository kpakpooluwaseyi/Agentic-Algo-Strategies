"""
Market Cipher A/B Money Flow Cross Strategy
============================================
A trend-following strategy based on the confluence of Market Cipher B's money flow
and wave signals, with an added momentum confirmation. This implementation adheres
to the strict MoonDev strategy development guidelines.

**Strategy Logic:**

Long Entry:
1.  **Trend Filter:** Price is above the 4-hour EMA(200).
2.  **Volume Confirmation:** Current 15m volume is above its 20-period SMA.
3.  **Money Flow:** VuManchu `rsimfi` indicator crosses above the zero line.
4.  **Cipher B Wave:** A `buy_signal` (green dot) appears.
5.  **Momentum:** MACD histogram is positive.

Short Entry:
1.  **Trend Filter:** Price is below the 4-hour EMA(200).
2.  **Volume Confirmation:** Current 15m volume is above its 20-period SMA.
3.  **Money Flow:** VuManchu `rsimfi` indicator crosses below the zero line.
4.  **Cipher B Wave:** A `sell_signal` (red dot) appears.
5.  **Momentum:** MACD histogram is negative.

**Risk Management (Mandatory):**
- **Stop Loss:** 2x ATR below entry (longs) or above entry (shorts).
- **Take Profit:** 3x ATR above entry (longs) or below entry (shorts).

**NOTE:** "Market Cipher A (Momentum Waves)" is not available in the provided `vumanchu`
library. As a proxy, this strategy uses a standard MACD histogram to confirm momentum,
adhering to the "Pure Math Only" and dependency constraints.
"""

import pandas as pd
import pandas_ta as ta
from backtesting import Strategy
from backtesting.lib import crossover

from src.indicators.vumanchu import cipher_b


def preprocess_data(df: pd.DataFrame, htf_ema_period=200, volume_sma_period=20,
                    macd_fast=12, macd_slow=26, macd_signal=9, atr_period=14) -> pd.DataFrame:
    """
    Adds all necessary indicators to the DataFrame for the strategy.
    """
    df = df.copy()

    # -- Mandatory MoonDev Guidelines --

    # 1. Higher Timeframe (HTF) Trend Filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema'] = ta.ema(df_4h['Close'], length=htf_ema_period)
    df_4h['htf_trend_up'] = (df_4h['Close'] > df_4h['ema']).astype(int)

    # Map HTF trend back to the original timeframe
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')
    df['htf_trend_up'].fillna(method='ffill', inplace=True) # Fill initial NaNs

    # 2. Volume Confirmation
    df['volume_sma'] = ta.sma(df['Volume'], length=volume_sma_period)

    # 3. ATR for Risk Management
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=atr_period)

    # -- Strategy Specific Indicators --

    # Market Cipher B (Money Flow and Waves/Dots)
    df = cipher_b(df)

    # Market Cipher A (Momentum Waves) - Proxy with MACD
    macd = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df['macd_hist'] = macd[f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}']

    # Convert boolean signals to int for backtesting.py
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    df.dropna(inplace=True)
    return df


class MarketCipherABMoneyFlowCross(Strategy):
    """
    Implements the Market Cipher A/B Money Flow Cross strategy with MoonDev guidelines.
    """
    # Optimizable parameters for risk management and filters, per guidelines
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_sma_period = 20
    htf_ema_period = 200

    # MACD parameters (proxy for Market Cipher A)
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    def init(self):
        """
        Initialize indicators using self.I() to have them available in next().
        """
        # --- MoonDev Guideline Indicators ---
        self.htf_trend_up = self.I(lambda: self.data.df['htf_trend_up'], name="htf_trend_up")
        self.volume_sma = self.I(lambda: self.data.df['volume_sma'], name="volume_sma")
        self.atr = self.I(lambda: self.data.df['atr'], name="atr")

        # --- Strategy-Specific Indicators ---
        # Market Cipher B
        self.money_flow = self.I(lambda: self.data.df['rsimfi'], name="money_flow")
        self.buy_dot = self.I(lambda: self.data.df['buy_signal'], name="buy_dot")
        self.sell_dot = self.I(lambda: self.data.df['sell_signal'], name="sell_dot")

        # Market Cipher A (MACD Proxy)
        self.macd_hist = self.I(lambda: self.data.df['macd_hist'], name="macd_hist")

    def next(self):
        """
        Define the strategy's trading logic.
        """
        price = self.data.Close[-1]

        # --- Pre-computation for Signals ---
        # Money flow must cross the zero line
        money_flow_cross_up = crossover(self.money_flow, 0)
        money_flow_cross_down = crossover(0, self.money_flow)

        # --- Filters ---
        # If a trade is already open, don't do anything
        if self.position:
            return

        # --- Long Entry Conditions ---
        if self.htf_trend_up[-1] == 1:
            if (self.data.Volume[-1] > self.volume_sma[-1]):
                if money_flow_cross_up:
                    if self.buy_dot[-1] == 1:
                        if self.macd_hist[-1] > 0:
                            # All conditions met, place a long trade
                            sl = price - self.atr[-1] * self.atr_sl_multiplier
                            tp = price + self.atr[-1] * self.atr_tp_multiplier
                            self.buy(sl=sl, tp=tp)

        # --- Short Entry Conditions ---
        if self.htf_trend_up[-1] == 0:
            if (self.data.Volume[-1] > self.volume_sma[-1]):
                if money_flow_cross_down:
                    if self.sell_dot[-1] == 1:
                        if self.macd_hist[-1] < 0:
                            # All conditions met, place a short trade
                            sl = price + self.atr[-1] * self.atr_sl_multiplier
                            tp = price - self.atr[-1] * self.atr_tp_multiplier
                            self.sell(sl=sl, tp=tp)
