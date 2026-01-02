import pandas as pd
import pandas_ta as ta

def wavetrend(high, low, close, channel_length=10, avg_length=21):
    """
    Calculates the WaveTrend Oscillator.
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    hlc3 = (high + low + close) / 3
    esa = ta.ema(hlc3, length=channel_length)
    esa = esa.bfill()
    d = ta.ema(abs(hlc3 - esa), length=channel_length)
    d = d.bfill()
    ci = (hlc3 - esa) / (0.015 * d)
    wt1 = ta.ema(ci, length=avg_length)
    wt1 = wt1.bfill()
    wt2 = ta.sma(wt1, length=4)
    wt2 = wt2.bfill()
    return wt1.values, wt2.values

def market_cipher_b(open, high, low, close, volume):
    """
    Calculates the Market Cipher B indicators (Momentum and Money Flow)
    and generates trading signals (dots).
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    volume = pd.Series(volume)

    # Momentum (WaveTrend Oscillator)
    wt1_series, wt2_series = pd.Series(wavetrend(high, low, close)[0]), pd.Series(wavetrend(high, low, close)[1])

    # Money Flow (Money Flow Index)
    mfi = ta.mfi(high, low, close, volume, length=14)
    mfi = mfi.bfill()

    # Signals (Dots)
    # Green dot: WT1 crosses above WT2
    green_dot = (wt1_series.shift(1) < wt2_series.shift(1)) & (wt1_series > wt2_series)
    # Red dot: WT1 crosses below WT2
    red_dot = (wt1_series.shift(1) > wt2_series.shift(1)) & (wt1_series < wt2_series)

    # For backtesting.py, convert boolean to int
    green_dot = green_dot.astype(int)
    red_dot = red_dot.astype(int)

    return wt1_series.values, wt2_series.values, mfi.values, green_dot.values, red_dot.values
