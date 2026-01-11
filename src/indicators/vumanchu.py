"""
VuManchu Cipher B Indicator - Python Port
==========================================
Ported from the original TradingView Pine Script.

This module provides:
- wavetrend(): The core WaveTrend oscillator (wt1, wt2)
- rsimfi(): RSI + Money Flow Index composite
- smma(): Smoothed Moving Average
- stoch_rsi(): Stochastic RSI

Usage:
    from src.indicators.vumanchu import wavetrend, rsimfi
    
    df = wavetrend(df)  # Adds wt1, wt2, wt_cross, buy_signal, sell_signal columns
    df = rsimfi(df)     # Adds rsimfi column
"""

import pandas as pd
import numpy as np


def smma(series: pd.Series, length: int) -> pd.Series:
    """
    Smoothed Moving Average (SMMA).
    Pine: smma := na(smma[1]) ? sma(src, len) : (smma[1] * (len - 1) + src) / len
    """
    # Find first valid index (non-NaN)
    first_valid = series.first_valid_index()
    if first_valid is None:
        return pd.Series(np.nan, index=series.index)
    
    first_valid_idx = series.index.get_loc(first_valid)
    smma_values = pd.Series(index=series.index, dtype=float)
    
    # Check if we have enough data after the first valid index
    if first_valid_idx + length > len(series):
         return pd.Series(np.nan, index=series.index)
         
    # Initialize with SMA of the first 'length' bars starting from first valid
    start_idx = first_valid_idx + length
    initial_sma = series.iloc[first_valid_idx:start_idx].mean()
    smma_values.iloc[start_idx - 1] = initial_sma
    
    # Calculate SMMA for the rest
    for i in range(start_idx, len(series)):
        smma_values.iloc[i] = (smma_values.iloc[i - 1] * (length - 1) + series.iloc[i]) / length
    
    return smma_values


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to be capitalized."""
    df = df.copy()
    df.columns = [str(col).strip().capitalize() for col in df.columns]
    return df


def rsimfi(df: pd.DataFrame, period: int = 60, multiplier: float = 230, pos_y: float = 2.5) -> pd.DataFrame:
    """
    RSI + Money Flow Index composite.
    Pine: mf = sma(((close - open) / (high - low)) * multiplier, period) - pos_y
          smma(mf, 4)
    
    Adds column: 'rsimfi'
    """
    df = _sanitize_columns(df)
    
    # Avoid division by zero
    hl_range = df['High'] - df['Low']
    hl_range = hl_range.replace(0, np.nan)
    
    mf_raw = ((df['Close'] - df['Open']) / hl_range) * multiplier
    mf = mf_raw.rolling(window=period).mean() - pos_y
    df['rsimfi'] = smma(mf, 4)
    
    return df


def wavetrend(df: pd.DataFrame, 
              channel_len: int = 9, 
              average_len: int = 12, 
              ma_len: int = 3,
              ob_level: int = 53,
              os_level: int = -53) -> pd.DataFrame:
    """
    WaveTrend Oscillator.
    
    Pine Logic:
        esa = ema(hlc3, chlen)
        de = ema(abs(hlc3 - esa), chlen)
        ci = (hlc3 - esa) / (0.015 * de)
        wt1 = ema(ci, avg)
        wt2 = sma(wt1, malen)
    
    Adds columns: 'wt1', 'wt2', 'wt_vwap', 'wt_cross', 'wt_cross_up', 'wt_cross_down',
                  'wt_oversold', 'wt_overbought', 'buy_signal', 'sell_signal'
    """
    df = _sanitize_columns(df)
    
    # hlc3 = (high + low + close) / 3
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
    
    # esa = EMA of hlc3
    esa = hlc3.ewm(span=channel_len, adjust=False).mean()
    
    # de = EMA of abs(hlc3 - esa)
    de = (hlc3 - esa).abs().ewm(span=channel_len, adjust=False).mean()
    
    # ci = (hlc3 - esa) / (0.015 * de)
    # Avoid division by zero
    ci = (hlc3 - esa) / (0.015 * de.replace(0, np.nan))
    
    # wt1 = EMA of ci
    wt1 = ci.ewm(span=average_len, adjust=False).mean()
    
    # wt2 = SMA of wt1
    wt2 = wt1.rolling(window=ma_len).mean()
    
    # VWAP difference
    wt_vwap = wt1 - wt2
    
    # Cross detection
    wt_cross = ((wt1.shift(1) < wt2.shift(1)) & (wt1 >= wt2)) | \
               ((wt1.shift(1) > wt2.shift(1)) & (wt1 <= wt2))
    
    wt_cross_up = wt1 >= wt2
    wt_cross_down = wt1 <= wt2
    
    # Overbought/Oversold
    wt_oversold = (wt1 <= os_level) & (wt2 <= os_level)
    wt_overbought = (wt1 >= ob_level) & (wt2 >= ob_level)
    
    # Buy Signal: Cross while oversold
    buy_signal = wt_cross & wt_cross_up & wt_oversold
    
    # Sell Signal: Cross while overbought
    sell_signal = wt_cross & wt_cross_down & wt_overbought
    
    # Assign to dataframe
    df['wt1'] = wt1
    df['wt2'] = wt2
    df['wt_vwap'] = wt_vwap
    df['wt_cross'] = wt_cross
    df['wt_cross_up'] = wt_cross_up
    df['wt_cross_down'] = wt_cross_down
    df['wt_oversold'] = wt_oversold
    df['wt_overbought'] = wt_overbought
    df['buy_signal'] = buy_signal
    df['sell_signal'] = sell_signal
    
    return df


def stoch_rsi(df: pd.DataFrame, 
              rsi_len: int = 14,
              stoch_len: int = 14,
              smooth_k: int = 3,
              smooth_d: int = 3) -> pd.DataFrame:
    """
    Stochastic RSI.
    
    Adds columns: 'stoch_rsi_k', 'stoch_rsi_d'
    """
    df = _sanitize_columns(df)
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_min = rsi.rolling(window=stoch_len).min()
    rsi_max = rsi.rolling(window=stoch_len).max()
    stoch_rsi_raw = 100 * (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    
    # Smooth K and D
    df['stoch_rsi_k'] = stoch_rsi_raw.rolling(window=smooth_k).mean()
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=smooth_d).mean()
    
    return df


def cipher_b(df: pd.DataFrame, 
             wt_channel_len: int = 9,
             wt_average_len: int = 12,
             wt_ma_len: int = 3,
             mfi_period: int = 60,
             mfi_multiplier: float = 230) -> pd.DataFrame:
    """
    Complete VuManchu Cipher B indicator suite.
    
    Combines all indicators into one dataframe:
    - WaveTrend (wt1, wt2, signals)
    - RSI+MFI
    - Stochastic RSI
    
    Returns dataframe with all indicator columns added.
    """
    df = df.copy()
    
    # Add WaveTrend
    df = wavetrend(df, wt_channel_len, wt_average_len, wt_ma_len)
    
    # Add RSI+MFI
    df = rsimfi(df, mfi_period, mfi_multiplier)
    
    # Add Stochastic RSI
    df = stoch_rsi(df)
    
    return df


# Test the indicators if run directly
if __name__ == '__main__':
    import pandas as pd
    
    # Load sample data
    try:
        df = pd.read_csv('data/BTC_1h.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("No data file found at data/BTC_1h.csv")
        exit(1)
    
    # Apply Cipher B indicators
    df = cipher_b(df)
    
    # Show results
    print("\n=== VuManchu Cipher B Indicators ===")
    print(df[['Close', 'wt1', 'wt2', 'rsimfi', 'buy_signal', 'sell_signal']].tail(20))
    
    # Count signals
    print(f"\nBuy signals: {df['buy_signal'].sum()}")
    print(f"Sell signals: {df['sell_signal'].sum()}")
