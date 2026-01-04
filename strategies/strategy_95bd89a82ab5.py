import pandas as pd
import talib
from backtesting import Strategy
from backtesting.lib import crossover

def preprocess_data(df: pd.DataFrame, **params):
    """
    Adds all indicators and pre-processing steps to the DataFrame,
    in compliance with the strategy development guidelines.
    """
    # Ensure index is a DatetimeIndex and name it 'datetime'
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = 'datetime'

    # Higher-Timeframe Trend Filter (4H EMA)
    df_4h = df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    if len(df_4h) > 200:
        df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
        df_4h['htf_trend_up'] = df_4h['Close'] > df_4h['ema_200']

        # Map the HTF trend back to the main DataFrame's index
        df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')

        # Back-fill to handle initial NaNs, then forward-fill again for safety
        df['htf_trend_up'] = df['htf_trend_up'].bfill().ffill()
    else:
        df['htf_trend_up'] = True

    # ATR and Volume MA
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # Adaptive Opening Range (first 15m candle of the UTC day)
    df['open_range_high'] = df.groupby(df.index.date)['High'].transform('first')
    df['open_range_low'] = df.groupby(df.index.date)['Low'].transform('first')

    # Let backtesting.py handle initial NaN values from indicators
    # df.dropna(inplace=True)
    return df

class OneCandleSetupFVG(Strategy):
    """
    A compliant implementation of the "One Candle Setup" strategy, adapted for 15m data.

    This strategy is an adaptation of a 1m/5m scalping strategy to the available
    15-minute BTC-USD data. It adheres strictly to the repository's mandatory
    development guidelines, including ATR-based risk management and a higher-timeframe
    trend filter.

    The core logic is as follows:
    1. Identify the high and low of the first 15m candle of each UTC day (the "opening range").
    2. Wait for a 15m candle to close outside this range, signaling a breakout.
    3. After the breakout, look for a Fair Value Gap (FVG) in the direction of the move.
    4. Enter a trade on the close of the FVG's third candle, provided the move
       is confirmed by the 4-hour trend and has sufficient volume.
    5. Set stop-loss and take-profit based on ATR multiples.
    """
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_multiplier = 1.0

    def init(self):
        # Pre-calculated indicators
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up)
        self.atr = self.I(lambda: self.data.atr)
        self.volume_ma = self.I(lambda: self.data.volume_ma)
        self.open_range_high = self.I(lambda: self.data.open_range_high)
        self.open_range_low = self.I(lambda: self.data.open_range_low)

        # State variables
        self.breakout_direction = 0
        self.current_day = None

    def next(self):
        # Daily state reset
        if self.current_day != self.data.index[-1].date():
            self.current_day = self.data.index[-1].date()
            self.breakout_direction = 0

        # Only trade if there is no open position
        if self.position:
            return

        # State 1: Wait for an opening range breakout
        if self.breakout_direction == 0:
            if self.data.Close[-1] > self.open_range_high[-1]:
                self.breakout_direction = 1  # Bullish breakout
            elif self.data.Close[-1] < self.open_range_low[-1]:
                self.breakout_direction = -1  # Bearish breakout
            return

        # State 2: Wait for a Fair Value Gap after breakout
        if len(self.data) < 3:
            return

        is_bullish_fvg = self.data.Low[-1] > self.data.High[-3]
        is_bearish_fvg = self.data.High[-1] < self.data.Low[-3]

        # Bullish Entry Logic
        if self.breakout_direction == 1 and is_bullish_fvg:
            # Trend and Volume Filters
            if not self.htf_trend_up[-1]:
                return
            if self.data.Volume[-1] < self.volume_ma[-1] * self.volume_ma_multiplier:
                return

            # Calculate SL and TP based on ATR
            entry_price = self.data.Close[-1]
            sl = entry_price - (self.atr_sl_multiplier * self.atr[-1])
            tp = entry_price + (self.atr_tp_multiplier * self.atr[-1])

            if tp > entry_price and entry_price > sl:
                self.buy(sl=sl, tp=tp)
                self.breakout_direction = 0 # Reset after taking trade

        # Bearish Entry Logic
        elif self.breakout_direction == -1 and is_bearish_fvg:
            # Trend and Volume Filters (short if not in HTF uptrend)
            if self.htf_trend_up[-1]:
                return
            if self.data.Volume[-1] < self.volume_ma[-1] * self.volume_ma_multiplier:
                return

            # Calculate SL and TP based on ATR
            entry_price = self.data.Close[-1]
            sl = entry_price + (self.atr_sl_multiplier * self.atr[-1])
            tp = entry_price - (self.atr_tp_multiplier * self.atr[-1])

            if tp < entry_price and entry_price < sl:
                self.sell(sl=sl, tp=tp)
                self.breakout_direction = 0 # Reset after taking trade

if __name__ == '__main__':
    from backtesting import Backtest
    import os
    import json

    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    results_file = os.path.join(results_dir, 'temp_result.json')
    plot_file = os.path.join(results_dir, 'strategy_95bd89a82ab5.html')

    # --- Data Loading and Preprocessing ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(
        data_path,
        parse_dates=['datetime'],
        index_col='datetime'
    )

    # Sanitize and capitalize column names for backtesting.py compatibility
    data.columns = [col.strip().title() for col in data.columns]

    processed_data = preprocess_data(data)

    # --- Backtesting ---
    bt = Backtest(processed_data, OneCandleSetupFVG, cash=10000, commission=.002)
    stats = bt.run()

    # --- Results ---
    print(stats)

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Save stats to JSON
    # A simple way to handle non-serializable types is to convert them to strings
    serializable_stats = {k: str(v) for k, v in stats.items() if not k.startswith('_')}
    with open(results_file, 'w') as f:
        json.dump(serializable_stats, f, indent=4)

    # Generate and save the plot
    try:
        bt.plot(filename=plot_file, open_browser=False)
        print(f"Plot saved to {plot_file}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
