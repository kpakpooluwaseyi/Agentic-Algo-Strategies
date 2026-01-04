"""
Bollinger Bands Breakout Strategy
"""
from backtesting import Strategy, Backtest
import pandas as pd
import pandas_ta as ta
import numpy as np

# Define constants for Bollinger Bands column names
BB_LEN = 20
BB_STD = 2.0
BB_UPPER_COL = 'BBU'
BB_LOWER_COL = 'BBL'
BB_MID_COL = 'BBM'
ATR_PERIOD = 14
ATR_COL = f'ATRr_{ATR_PERIOD}'
SQUEEZE_LOOKBACK = 252

def preprocess_data(df, bb_length=BB_LEN, bb_std=BB_STD, atr_period=ATR_PERIOD, volume_ma_period=20, htf_period=50, squeeze_lookback=SQUEEZE_LOOKBACK):
    """
    Adds all the required indicators to the dataframe.
    """
    df.ta.bbands(length=bb_length, std=bb_std, append=True)

    # Dynamically find and rename bbands columns for consistency
    try:
        bbu_col_dyn = [col for col in df.columns if col.startswith(f'BBU_{bb_length}_')][0]
        bbl_col_dyn = [col for col in df.columns if col.startswith(f'BBL_{bb_length}_')][0]
        bbm_col_dyn = [col for col in df.columns if col.startswith(f'BBM_{bb_length}_')][0]
        df.rename(columns={
            bbu_col_dyn: BB_UPPER_COL,
            bbl_col_dyn: BB_LOWER_COL,
            bbm_col_dyn: BB_MID_COL,
        }, inplace=True)
    except IndexError:
        raise ValueError("Could not find Bollinger Bands columns. Check pandas_ta version or column naming.")

    # Bollinger Band Width
    df['bbw'] = (df[BB_UPPER_COL] - df[BB_LOWER_COL]) / df[BB_MID_COL]

    # Rolling percentile rank for squeeze detection (no lookahead bias)
    df['bbw_rank'] = df['bbw'].rolling(window=squeeze_lookback).rank(pct=True)

    # ATR for risk management
    df.ta.atr(length=atr_period, append=True)

    # Volume MA for confirmation
    df['volume_ma'] = df['Volume'].rolling(window=volume_ma_period).mean()

    # Higher timeframe trend filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).copy()

    df_4h['ema_htf'] = ta.ema(df_4h['Close'], length=htf_period)
    df_4h['htf_trend_up'] = (df_4h['Close'] > df_4h['ema_htf']).astype(int)

    # Forward-fill the 4H trend onto the 15m data
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')

    df.dropna(inplace=True)

    return df


class BollingerBandsBreakout(Strategy):
    """
    Implements the Bollinger Bands Breakout strategy.
    """
    # Optimizable parameters
    squeeze_threshold_prct = 0.1
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        """
        Initializes the strategy.
        """
        self.bb_upper = self.I(lambda: self.data.df[BB_UPPER_COL], name="BB_Upper")
        self.bb_lower = self.I(lambda: self.data.df[BB_LOWER_COL], name="BB_Lower")
        self.bbw_rank = self.I(lambda: self.data.df['bbw_rank'], name="BBW_Rank")
        self.atr = self.I(lambda: self.data.df[ATR_COL], name="ATR")
        self.volume_ma = self.I(lambda: self.data.df['volume_ma'], name="Volume_MA")
        self.htf_trend_up = self.I(lambda: self.data.df['htf_trend_up'], name="HTF_Trend_Up")

        self.in_squeeze = False

    def next(self):
        """
        Defines the trading logic.
        """
        # If we are already in a position, do nothing.
        if self.position:
            return

        # Squeeze Detection: If bands are narrow, enter the "squeeze" state.
        if not self.in_squeeze and self.bbw_rank[-1] < self.squeeze_threshold_prct:
            self.in_squeeze = True
            return # Wait for the next bar to check for a breakout

        # If not in a squeeze state, do nothing.
        if not self.in_squeeze:
            return

        # --- Breakout Conditions ---
        # (Only checked if `self.in_squeeze` is True)

        volume_conf = self.data.Volume[-1] > self.volume_ma[-1]
        breakout_candle_len = self.data.High[-1] - self.data.Low[-1]
        prev_candle_len = self.data.High[-2] - self.data.Low[-2]
        is_strong_candle = breakout_candle_len > prev_candle_len

        # Long Breakout
        long_breakout = self.data.Close[-1] > self.bb_upper[-1]
        if self.htf_trend_up[-1] and long_breakout and volume_conf and is_strong_candle:
            sl = self.data.Close[-1] - self.atr[-1] * self.atr_sl_multiplier
            tp = self.data.Close[-1] + self.atr[-1] * self.atr_tp_multiplier
            self.buy(sl=sl, tp=tp)
            self.in_squeeze = False # Exit squeeze state after taking the trade

        # Short Breakout
        short_breakout = self.data.Close[-1] < self.bb_lower[-1]
        if not self.htf_trend_up[-1] and short_breakout and volume_conf and is_strong_candle:
            sl = self.data.Close[-1] + self.atr[-1] * self.atr_sl_multiplier
            tp = self.data.Close[-1] - self.atr[-1] * self.atr_tp_multiplier
            self.sell(sl=sl, tp=tp)
            self.in_squeeze = False # Exit squeeze state after taking the trade


if __name__ == '__main__':
    try:
        # Robust data loading
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
        df.columns = [col.strip().capitalize() for col in df.columns]
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain the following columns: {required_cols}")
        df = df[required_cols]
        df.index.name = 'datetime'
    except FileNotFoundError:
        print("Data file not found. A sample dataset will be generated.")
        data = np.random.randn(5000, 5) * np.array([1, 0.01, 0.01, 0.01, 100])
        data[:, 0] = 50000 + np.cumsum(data[:, 0])
        data[:, 4] = np.abs(data[:, 4])
        df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                          index=pd.to_datetime(pd.date_range('2020-01-01', periods=5000, freq='15min')))
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(2)
        df['Volume'] = df['Volume'].round(2)

    df_processed = preprocess_data(df)
    bt = Backtest(df_processed, BollingerBandsBreakout, cash=100000, commission=.002)
    stats = bt.run()
    print(stats)

    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    try:
        bt.plot(filename='results/strategy_576ebdbc1dd6.html')
    except Exception as e:
        print(f"Could not generate plot: {e}")

    import json
    # Sanitize stats for JSON serialization
    stats_serializable = dict(stats)

    # Remove non-serializable items
    stats_serializable.pop('_strategy', None)
    stats_serializable.pop('_equity_curve', None)
    stats_serializable.pop('_trades', None)

    for key, value in stats_serializable.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_serializable[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
             stats_serializable[key] = float(value)

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_serializable, f, indent=4)
