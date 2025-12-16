import json
import os
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data and adds the previous day's ATR.
    """
    df.columns = [x.strip().capitalize() for x in df.columns]
    if 'Unnamed: 6' in df.columns:
        df.drop(columns=['Unnamed: 6'], inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    daily_df = df.resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last'})
    daily_df['ATR'] = ta.atr(daily_df['High'], daily_df['Low'], daily_df['Close'], length=14)
    daily_df['Prev_Day_ATR'] = daily_df['ATR'].shift(1)
    df['date'] = df.index.date
    atr_map = daily_df.set_index(daily_df.index.date)['Prev_Day_ATR']
    df['Prev_Day_ATR'] = df['date'].map(atr_map)
    df.drop(columns=['date'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    return df

def is_bullish_engulfing(data):
    if len(data.Open) < 2: return False
    return (data.Open[-2] > data.Close[-2] and data.Close[-1] > data.Open[-1] and
            data.Close[-1] >= data.Open[-2] and data.Open[-1] <= data.Close[-2])

def is_bearish_engulfing(data):
    if len(data.Open) < 2: return False
    return (data.Open[-2] < data.Close[-2] and data.Close[-1] < data.Open[-1] and
            data.Close[-1] <= data.Open[-2] and data.Open[-1] >= data.Close[-2])

def is_hammer(data, target_wick_ratio=2, max_body_ratio=0.3):
    body = abs(data.Open[-1] - data.Close[-1])
    candle_range = data.High[-1] - data.Low[-1]
    if candle_range == 0: return False
    lower_wick = min(data.Open[-1], data.Close[-1]) - data.Low[-1]
    upper_wick = data.High[-1] - max(data.Open[-1], data.Close[-1])
    return (lower_wick >= target_wick_ratio * body and upper_wick / candle_range <= max_body_ratio)

def is_inverted_hammer(data, target_wick_ratio=2, max_body_ratio=0.3):
    body = abs(data.Open[-1] - data.Close[-1])
    candle_range = data.High[-1] - data.Low[-1]
    if candle_range == 0: return False
    upper_wick = data.High[-1] - max(data.Open[-1], data.Close[-1])
    lower_wick = min(data.Open[-1], data.Close[-1]) - data.Low[-1]
    return (upper_wick >= target_wick_ratio * body and lower_wick / candle_range <= max_body_ratio)

def sanitize_stats(stats):
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (np.integer, np.int64)): sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)): sanitized[key] = float(value) if not np.isnan(value) else None
        elif isinstance(value, pd.Timestamp): sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Series): sanitized[key] = value.to_dict()
        else: sanitized[key] = value
    return sanitized

class QuickFlipScalperStrategy(Strategy):
    atr_threshold_pct = 25
    sl_buffer_pct = 1.0
    entry_window_minutes = 90

    def init(self):
        self.opening_range_high = None
        self.opening_range_low = None
        self.is_opening_candle_validated = False
        self.trade_taken_today = False
        self.current_day = None

    def next(self):
        current_time = self.data.index[-1]

        if self.current_day != current_time.date():
            self.current_day = current_time.date()
            self.opening_range_high = None
            self.opening_range_low = None
            self.is_opening_candle_validated = False
            self.trade_taken_today = False

        if self.trade_taken_today or self.position: return

        market_open_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        entry_window_end = market_open_time + pd.Timedelta(minutes=self.entry_window_minutes)

        if not (market_open_time <= current_time < entry_window_end): return

        if self.opening_range_high is None and current_time.time() >= pd.Timestamp('00:15').time():
            opening_candle_idx = self.data.index.searchsorted(market_open_time)
            self.opening_range_high = self.data.High[opening_candle_idx]
            self.opening_range_low = self.data.Low[opening_candle_idx]

            opening_candle_range = self.opening_range_high - self.opening_range_low
            atr_check_value = self.data.Prev_Day_ATR[-1] * (self.atr_threshold_pct / 100)

            if opening_candle_range >= atr_check_value:
                self.is_opening_candle_validated = True

        if not self.is_opening_candle_validated: return

        opening_candle_idx = self.data.index.searchsorted(market_open_time)
        opening_candle_open = self.data.Open[opening_candle_idx]
        opening_candle_close = self.data.Close[opening_candle_idx]
        opening_candle_was_bullish = opening_candle_close > opening_candle_open

        if opening_candle_was_bullish and self.data.High[-1] > self.opening_range_high:
            if is_bearish_engulfing(self.data) or is_inverted_hammer(self.data):
                sl = self.data.High[-1] * (1 + self.sl_buffer_pct / 100); tp = self.opening_range_low
                if self.data.Close[-1] > tp:
                    self.sell(sl=sl, tp=tp); self.trade_taken_today = True

        elif not opening_candle_was_bullish and self.data.Low[-1] < self.opening_range_low:
            if is_bullish_engulfing(self.data) or is_hammer(self.data):
                sl = self.data.Low[-1] * (1 - self.sl_buffer_pct / 100); tp = self.opening_range_high
                if self.data.Close[-1] < tp:
                    self.buy(sl=sl, tp=tp); self.trade_taken_today = True

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path): raise FileNotFoundError(f"Data not found at {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data = preprocess_data(data)

    bt = Backtest(data, QuickFlipScalperStrategy, cash=100_000, commission=.002)
    stats = bt.run()

    print("Backtest Results:"); print(stats)
    os.makedirs('results', exist_ok=True)
    results_dict = sanitize_stats(stats.to_dict())
    final_results = {'strategy_name':'quick_flip_scalper', 'return': results_dict.get('Return [%]'), 'sharpe': results_dict.get('Sharpe Ratio'),
                     'max_drawdown': results_dict.get('Max. Drawdown [%]'), 'win_rate': results_dict.get('Win Rate [%]'), 'total_trades': results_dict.get('# Trades')}
    with open('results/temp_result.json', 'w') as f: json.dump(final_results, f, indent=4); f.write('\n')
    print("\nResults saved to results/temp_result.json")
    plot_filename = 'results/quick_flip_scalper.html'
    try: bt.plot(filename=plot_filename, open_browser=False); print(f"Plot saved to {plot_filename}")
    except Exception as e: print(f"\nCould not generate plot: {e}")
