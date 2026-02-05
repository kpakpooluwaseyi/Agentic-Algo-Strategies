import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest
import os
import json

# State constants
SEARCHING = 0
MONITORING_10AM_CANDLE = 1
WAITING_FOR_RETRACEMENT = 2

def sanitize_stats(stats):
    sanitized = {}
    if stats is None: return {}
    # Use ._asdict() for namedtuples, else .to_dict() for Series, else the dict itself
    stats_dict = stats._asdict() if hasattr(stats, '_asdict') else stats.to_dict() if isinstance(stats, pd.Series) else stats

    for key, value in stats_dict.items():
        if key in ['_equity_curve', '_trades', '_strategy']:
            continue
        if pd.isna(value):
            sanitized[key] = None
            continue
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value)
        else:
            sanitized[key] = value
    return sanitized

def preprocess_data(df, **params):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # Make timezone aware if it's not already
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC', nonexistent='shift_forward').tz_convert('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')

    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['volume_ma'] = df['Volume'].rolling(20).mean()

    df_d = df.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    df_d['ema_50'] = talib.EMA(df_d['Close'], timeperiod=50)
    df_d['htf_uptrend'] = (df_d['Close'] > df_d['ema_50']).astype(int)
    df['daily_uptrend'] = df.index.normalize().map(df_d['htf_uptrend'])

    # Use 'h' for frequency and offset parameter for timezone-aware resampling
    df_4h = df.resample('4h', offset='2h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

    daily_signals = {}
    for date_ts in df_4h.index.normalize().unique():
        date = date_ts.date()
        day_candles = df_4h[df_4h.index.date == date]
        candle_2am = day_candles[day_candles.index.hour == 2]
        candle_6am = day_candles[day_candles.index.hour == 6]

        if candle_2am.empty or candle_6am.empty: continue

        candle_2am, candle_6am = candle_2am.iloc[0], candle_6am.iloc[0]

        hand_tipped_bearish = (candle_6am['High'] > candle_2am['High']) and (candle_6am['Close'] < candle_2am['High'])
        hand_tipped_bullish = (candle_6am['Low'] < candle_2am['Low']) and (candle_6am['Close'] > candle_2am['Low'])

        day_15m = df[df.index.date == date]
        pre_market_range = day_15m.between_time('00:00', '09:29')
        manipulation_window = day_15m.between_time('09:30', '09:59')

        if pre_market_range.empty or manipulation_window.empty: continue

        setup_signal = 0
        if hand_tipped_bearish and (manipulation_window['High'].max() > pre_market_range['High'].max()): setup_signal = -1
        if hand_tipped_bullish and (manipulation_window['Low'].min() < pre_market_range['Low'].min()): setup_signal = 1
        daily_signals[date] = setup_signal

    df['setup_signal'] = pd.Series(df.index.date, index=df.index).map(daily_signals).fillna(0)

    # More robust FVG detection: displacement candle should be strong
    body_threshold = 0.7
    df['body'] = abs(df['Close'] - df['Open'])
    df['is_strong_displacement'] = df['body'] > (df['atr'] * body_threshold)

    # FVG is formed by a strong candle at shift(1), so check for displacement there
    df['has_bearish_fvg'] = (df['Low'].shift(2) > df['High']) & df['is_strong_displacement'].shift(1)
    df['has_bullish_fvg'] = (df['High'].shift(2) < df['Low']) & df['is_strong_displacement'].shift(1)

    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def generate_synthetic_data():
    """Generates synthetic data with a perfect textbook pattern."""
    ny = 'America/New_York'
    base_time = pd.Timestamp('2023-10-26 00:00', tz=ny)

    # Create a day's worth of 15m data
    index = pd.date_range(start=base_time, periods=96, freq='15min')
    data = pd.DataFrame(index=index)

    # Base price and volume
    data['Open'] = 100
    data['High'] = 100
    data['Low'] = 100
    data['Close'] = 100
    data['Volume'] = 1000

    # 2:00 4H Candle (consolidation)
    data.loc[data.index.hour == 2, 'High'] = 101
    data.loc[data.index.hour == 2, 'Low'] = 99

    # 6:00 4H Candle (manipulation)
    data.loc['2023-10-26 06:00':'2023-10-26 09:45', 'Open'] = 101
    data.loc['2023-10-26 06:00':'2023-10-26 09:45', 'High'] = 102 # Sweeps 2am high
    data.loc['2023-10-26 06:00':'2023-10-26 09:45', 'Low'] = 100
    data.loc['2023-10-26 06:00':'2023-10-26 09:45', 'Close'] = 100.5 # Closes inside 2am range

    # 9:30 Manipulation
    data.loc['2023-10-26 09:30', 'High'] = 102.5 # Sweeps pre-market high

    # 10:00 4H Candle (distribution)
    data.loc['2023-10-26 10:00', ['Open', 'High', 'Low', 'Close']] = [100.5, 100.8, 98, 98.2]

    # Strong displacement candle creating the FVG
    data.loc['2023-10-26 10:15', ['Open', 'High', 'Low', 'Close']] = [98.2, 98.3, 96, 96.1]

    # Third candle, leaving the gap
    data.loc['2023-10-26 10:30', ['Open', 'High', 'Low', 'Close']] = [96.1, 97, 95.5, 96.5]

    # Fill rest of the day
    data = data.ffill()
    data['datetime'] = data.index.tz_convert('UTC')
    data.reset_index(drop=True, inplace=True)
    return data

class OneCandle6am930MarketMaker(Strategy):
    atr_sl_multiplier = 1.5
    atr_tp_multiplier = 3.0

    def init(self):
        self.atr = self.I(lambda: self.data.atr, name="atr")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="volume_ma")
        self.daily_uptrend = self.I(lambda: self.data.daily_uptrend, name="daily_uptrend")
        self.setup_signal = self.I(lambda: self.data.setup_signal, name="setup_signal")
        self.has_bearish_fvg = self.I(lambda: self.data.has_bearish_fvg, name="has_bearish_fvg")
        self.has_bullish_fvg = self.I(lambda: self.data.has_bullish_fvg, name="has_bullish_fvg")

        self.state = SEARCHING
        self.entry_window_high = 0.0
        self.entry_window_low = 0.0

    def next(self):
        if self.position: return

        current_time = self.data.index[-1].time()

        if self.state == MONITORING_10AM_CANDLE and not (10 <= current_time.hour < 14):
             self.state = SEARCHING

        if self.state == SEARCHING:
            if self.setup_signal[-1] != 0:
                self.state = MONITORING_10AM_CANDLE
                self.entry_window_high = 0.0
                self.entry_window_low = float('inf')

        if self.state == MONITORING_10AM_CANDLE:
            is_entry_window = 10 <= current_time.hour < 14
            if not is_entry_window: return

            if self.entry_window_high == 0.0:
                self.entry_window_high = self.data.High[-1]
                self.entry_window_low = self.data.Low[-1]
            else:
                self.entry_window_high = max(self.entry_window_high, self.data.High[-1])
                self.entry_window_low = min(self.entry_window_low, self.data.Low[-1])

            if self.data.index[-1].hour == 10 and self.data.index[-1].minute < 30: return
            if self.data.Volume[-1] < self.volume_ma[-1]: return

            direction = self.setup_signal[-1]
            atr = self.atr[-1]

            if direction == -1 and not self.daily_uptrend[-1]:
                if self.data.Close[-1] < self.data.Open[-1] and self.has_bearish_fvg[-1]:
                    sl = self.entry_window_high + self.atr_sl_multiplier * atr
                    tp = self.data.Close[-1] - self.atr_tp_multiplier * atr
                    if tp < sl: self.sell(sl=sl, tp=tp)

            elif direction == 1 and self.daily_uptrend[-1]:
                if self.data.Close[-1] > self.data.Open[-1] and self.has_bullish_fvg[-1]:
                    sl = self.entry_window_low - self.atr_sl_multiplier * atr
                    tp = self.data.Close[-1] + self.atr_tp_multiplier * atr
                    if tp > sl: self.buy(sl=sl, tp=tp)

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # --- Run with Synthetic Data to Verify Logic ---
    print("--- Verifying logic with Synthetic Data ---")
    synthetic_data = generate_synthetic_data()
    processed_synthetic = preprocess_data(synthetic_data)

    if processed_synthetic.empty:
        print("Synthetic data processing failed.")
    else:
        bt_synth = Backtest(processed_synthetic, OneCandle6am930MarketMaker, cash=100000, commission=.002)
        stats_synth = bt_synth.run()
        print("\nSynthetic Backtest Results:")
        print(stats_synth)
        if stats_synth['# Trades'] > 0:
            print("✅ Logic verification successful: A trade was executed on synthetic data.")
        else:
            print("❌ Logic verification failed: No trade was executed on synthetic data.")

    # --- Run with Real Data ---
    print("\n--- Running backtest with Real Market Data (GOOG) ---")
    try:
        from backtesting.test import GOOG
        data = GOOG.iloc[-3000:].copy()
        data.columns = [col.title() for col in data.columns]
        data['datetime'] = data.index
    except ImportError:
        print("Falling back to BTC data. Note: Strategy is designed for market hours.")
        try: data = pd.read_csv('data/BTC-USD-15m.csv')
        except FileNotFoundError: exit("Error: No suitable data file found.")

    processed_data = preprocess_data(data)

    if processed_data.empty: exit("Processing resulted in empty data.")

    bt = Backtest(processed_data, OneCandle6am930MarketMaker, cash=100000, commission=.002)

    print("Running backtest with final time-based logic...")
    stats = bt.run()

    print("\nBacktest Results:")
    print(stats)

    json_path = 'results/temp_result.json'
    sanitized = sanitize_stats(stats)
    with open(json_path, 'w') as f: json.dump(sanitized, f, indent=4)
    print(f"\nResults saved to {json_path}")

    plot_path = 'results/one_candle_6am_930_market_maker.html'
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
