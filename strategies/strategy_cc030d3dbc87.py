"""
Strategy: Market Cipher B Trend Acceleration
Implementation: strategy_cc030d3dbc87
"""
import os
import sys
import pandas as pd
import talib

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Applies the necessary indicators and filters to the raw DataFrame.

    Args:
        df: The raw OHLCV DataFrame.
        **params: Additional parameters (unused in this version).

    Returns:
        The DataFrame with added indicators.
    """
    # 1. Apply Cipher B indicator
    # This adds 'wt1' (the blue wave) and other related signals.
    df = cipher_b(df)

    # 2. Add ATR for risk management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # 3. Add higher-timeframe (4H) trend filter
    # Resample to 4H timeframe to calculate the trend indicator
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
    }).dropna()

    # Calculate 200-period EMA on the 4H data
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)

    # Determine the higher-timeframe trend direction
    df_4h['htf_trend_up'] = (df_4h['Close'] > df_4h['ema_200']).astype(int)

    # Map the 4H trend back to the original 15m timeframe
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')

    # 4. Add Volume confirmation filter
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # Drop rows with NaN values resulting from indicator calculations
    df.dropna(inplace=True)

    return df
