
import pandas as pd
import numpy as np
import pandas_ta as ta

def create_processed_data():
    """
    Loads the raw BTC data, cleans it, calculates all necessary indicators,
    and saves it to a new CSV file.
    """
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv')
    except FileNotFoundError:
        print("Raw data file not found.")
        return

    # --- ROBUST DATA CLEANING ---
    df.columns = [col.strip().capitalize() for col in df.columns]
    if 'Unnamed: 6' in df.columns:
        df.drop(columns=['Unnamed: 6'], inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    # --- END ROBUST DATA CLEANING ---

    # --- INDICATOR CALCULATION ---
    ema_period = 200
    atr_period = 14
    atr_multiplier = 2.0
    volume_ma_period = 20

    df.ta.atr(high='High', low='Low', close='Close', length=atr_period, append=True)
    df.rename(columns={f'ATRr_{atr_period}': 'atr'}, inplace=True)

    df['ema'] = ta.ema(df['Close'], length=ema_period)
    df['upper_band'] = df['ema'] + (df['atr'] * atr_multiplier)
    df['lower_band'] = df['ema'] - (df['atr'] * atr_multiplier)

    df_4h = df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['htf_ema'] = ta.ema(df_4h['Close'], length=ema_period)
    df_4h['htf_trend'] = np.where(df_4h['Close'] > df_4h['htf_ema'], 1, -1)
    df['htf_trend'] = df_4h['htf_trend'].reindex(df.index, method='ffill').fillna(0)

    df['volume_ma'] = ta.sma(df['Volume'], length=volume_ma_period)

    df.dropna(inplace=True)
    # --- END INDICATOR CALCULATION ---

    # Save to a new file
    df.to_csv('data/processed_btc.csv')
    print("Processed data saved to data/processed_btc.csv")

if __name__ == '__main__':
    create_processed_data()
