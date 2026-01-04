import sys
import os
# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting import Strategy, Backtest
import pandas as pd
import pandas_ta as ta
from scipy.signal import find_peaks
import numpy as np

from src.indicators import vumanchu

def preprocess_data(df: pd.DataFrame, ema_fast_period=50, ema_slow_period=200, atr_period=14, volume_ma_period=20) -> pd.DataFrame:
    """
    Adds all necessary indicators to the DataFrame.
    """
    df = df.copy()

    # Standard Indicators
    df[f'EMA_{ema_fast_period}'] = ta.ema(df['Close'], length=ema_fast_period)
    df[f'EMA_{ema_slow_period}'] = ta.ema(df['Close'], length=ema_slow_period)
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=atr_period)
    df[f'volume_ma_{volume_ma_period}'] = ta.sma(df['Volume'], length=volume_ma_period)

    # VuManchu Cipher B
    df = vumanchu.cipher_b(df)

    # Multi-Timeframe Trend Filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = ta.ema(df_4h['Close'], length=200)
    df_4h['htf_trend_up'] = df_4h['Close'] > df_4h['ema_200']

    # Ensure the 4H index is unique before mapping
    df_4h.index = df_4h.index.normalize()
    df_4h = df_4h[~df_4h.index.duplicated(keep='first')]

    # Map 4H trend to original timeframe
    df['htf_trend_up'] = df.index.normalize().map(df_4h['htf_trend_up'])
    df['htf_trend_up'] = df['htf_trend_up'].bfill().ffill()

    # The backtesting framework handles NaNs, so a full drop is not needed and can clear the dataframe
    # df.dropna(inplace=True)
    return df


class VuManchuEmaCrossMoneyflow(Strategy):
    """
    Strategy based on VuManchu Cipher B, EMA crosses, and a multi-timeframe trend filter.
    """
    # Optimizable parameters
    ema_fast_period = 50
    ema_slow_period = 200
    atr_period = 14
    atr_sl_multiplier = 2.0
    risk_reward_ratio = 2.0
    volume_ma_period = 20

    def init(self):
        """
        Initialize the indicators.
        """
        # Indicators are pre-calculated in preprocess_data,
        # so we just create references to them here.
        self.ema_fast = self.I(lambda x: x, self.data.df[f'EMA_{self.ema_fast_period}'], name=f'EMA_{self.ema_fast_period}')
        self.ema_slow = self.I(lambda x: x, self.data.df[f'EMA_{self.ema_slow_period}'], name=f'EMA_{self.ema_slow_period}')
        self.rsimfi = self.I(lambda x: x, self.data.df['rsimfi'], name='rsimfi')
        self.wt1 = self.I(lambda x: x, self.data.df['wt1'], name='wt1')
        self.wt2 = self.I(lambda x: x, self.data.df['wt2'], name='wt2')
        self.htf_trend_up = self.I(lambda x: x, self.data.df['htf_trend_up'], name='htf_trend_up')
        self.atr = self.I(lambda x: x, self.data.df['atr'], name='atr')
        self.volume_ma = self.I(lambda x: x, self.data.df[f'volume_ma_{self.volume_ma_period}'], name=f'volume_ma_{self.volume_ma_period}')

        # Pre-calculate swing points for stop loss placement
        self.highs = self.data.High
        self.lows = self.data.Low

        # Using a larger distance to find more significant peaks/troughs
        self.swing_highs, _ = find_peaks(self.highs, distance=15)
        self.swing_lows, _ = find_peaks(-self.lows, distance=15)


    def next(self):
        """
        Define the entry and exit logic.
        """
        price = self.data.Close[-1]

        # Basic trend and volume conditions
        is_uptrend = price > self.ema_slow[-1] and self.htf_trend_up[-1]
        is_downtrend = price < self.ema_slow[-1] and not self.htf_trend_up[-1]
        has_volume = self.data.Volume[-1] > self.volume_ma[-1]

        # --- Entry Conditions ---

        # Long Entry
        if is_uptrend and has_volume and not self.position:
            # Price pulled back to 50 EMA
            if self.data.Low[-1] <= self.ema_fast[-1]:
                # Money flow is green
                if self.rsimfi[-1] > 0:
                    # Blue waves cross up from below zero
                    if self.wt1[-1] > self.wt2[-1] and self.wt1[-2] < self.wt2[-2] and self.wt1[-1] < 0:

                        # Find the most recent swing low for SL
                        relevant_lows = self.swing_lows[self.swing_lows < len(self.data.Close) -1]
                        if len(relevant_lows) > 0:
                            last_swing_low_idx = relevant_lows[-1]
                            last_swing_low_price = self.lows[last_swing_low_idx]

                            atr_val = self.atr[-1]
                            stop_loss = last_swing_low_price - (self.atr_sl_multiplier * atr_val)
                            take_profit = price + (price - stop_loss) * self.risk_reward_ratio

                            if take_profit > price and stop_loss < price:
                                self.buy(sl=stop_loss, tp=take_profit)

        # Short Entry
        if is_downtrend and has_volume and not self.position:
            # Price pulled back to 50 EMA
            if self.data.High[-1] >= self.ema_fast[-1]:
                 # Money flow is red
                if self.rsimfi[-1] < 0:
                    # Blue waves cross down from above zero
                    if self.wt1[-1] < self.wt2[-1] and self.wt1[-2] > self.wt2[-2] and self.wt1[-1] > 0:

                        # Find the most recent swing high for SL
                        relevant_highs = self.swing_highs[self.swing_highs < len(self.data.Close) -1]
                        if len(relevant_highs) > 0:
                            last_swing_high_idx = relevant_highs[-1]
                            last_swing_high_price = self.highs[last_swing_high_idx]

                            atr_val = self.atr[-1]
                            stop_loss = last_swing_high_price + (self.atr_sl_multiplier * atr_val)
                            take_profit = price - (stop_loss - price) * self.risk_reward_ratio

                            if take_profit < price and stop_loss > price:
                                self.sell(sl=stop_loss, tp=take_profit)

if __name__ == '__main__':
    # Load data
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        # Sanitize column names (lowercase, strip spaces)
        data.columns = [col.strip().lower() for col in data.columns]
        # Rename to required format
        data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure the data file is in the correct directory.")
        exit()

    # Preprocess data
    data = preprocess_data(data)

    # Run backtest
    bt = Backtest(data, VuManchuEmaCrossMoneyflow, cash=100_000, commission=.002)

    stats = bt.run()
    print(stats)

    import os
    os.makedirs('results', exist_ok=True)

    # Save results and plot
    # Sanitize the stats object for JSON serialization
    sanitized_stats = {key: (str(value) if isinstance(value, (pd.Timestamp, pd.Timedelta)) else value) for key, value in stats.items() if not isinstance(value, (pd.DataFrame, pd.Series))}
    sanitized_stats.pop('_strategy', None)
    sanitized_stats.pop('_equity_curve', None)
    sanitized_stats.pop('_trades', None)

    with open('results/temp_result.json', 'w') as f:
        import json
        json.dump(sanitized_stats, f, indent=4)

    try:
        bt.plot(filename='results/strategy_a5c83aab5aa8_plot.html', open_browser=False)
    except Exception as e:
        print(f"Error plotting: {e}")
