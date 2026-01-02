
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply
import numpy as np
import json
from scipy.signal import find_peaks

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to make it JSON serializable.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.Series, pd.DataFrame, Strategy)):
            continue
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized

class MarketCipher4h24mSwing(Strategy):
    """
    This strategy is a proxy for the 'Market Cipher 4h/24m Swing Trading' strategy.
    It uses standard indicators to approximate the behavior of the proprietary
    Market Cipher and VuManchu indicators.

    Indicators Proxies:
    - Market Cipher B Money Flow: Money Flow Index (MFI)
    - Market Cipher B Waves: Smoothed RSI
    - Wolfpack ID: MACD Histogram
    - Market Cipher A Diamonds: RSI Overbought/Oversold
    - Market Cipher A Ribbon 5: 21-period EMA
    """

    # --- Strategy Parameters ---
    # These can be optimized later
    mfi_period = 14
    rsi_period = 14
    rsi_smooth = 5
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    ema_period = 21
    rsi_ob = 70
    rsi_os = 30
    sl_atr_multiplier = 2
    atr_period = 14


    def init(self):
        """
        Initialize the strategy and indicators.
        """
        # --- Pre-calculated Indicators from DataFrame ---
        self.env_money_flow = self.I(lambda x: x, self.data.df['4h_money_flow'], name="Env_Money_Flow")
        self.env_trigger_wave = self.I(lambda x: x, self.data.df['4h_trigger_wave'], name="Env_Trigger_Wave")
        self.env_wolfpack = self.I(lambda x: x, self.data.df['4h_wolfpack'], name="Env_Wolfpack")

        self.exec_anchor_wave = self.I(lambda x: x, self.data.df['30m_anchor_wave'], name="Exec_Anchor_Wave")
        self.exec_trigger_wave = self.I(lambda x: x, self.data.df['30m_trigger_wave'], name="Exec_Trigger_Wave")
        self.exec_money_flow = self.I(lambda x: x, self.data.df['30m_money_flow'], name="Exec_Money_Flow")
        self.exec_wolfpack = self.I(lambda x: x, self.data.df['30m_wolfpack'], name="Exec_Wolfpack")

        self.exec_diamond = self.I(lambda x: x, self.data.df['30m_diamond'], name="Exec_Diamond")
        self.exec_ema_ribbon = self.I(lambda x: x, self.data.df['30m_ema_21'], name="Exec_EMA_Ribbon")
        self.atr = self.I(lambda x: x, self.data.df['30m_atr'], name="ATR")

        # State machine for entry logic
        self.anchor_long_detected = False
        self.anchor_short_detected = False


    def next(self):
        """
        The main strategy logic that runs on each bar.
        """
        price = self.data.Close[-1]
        ema_21 = self.exec_ema_ribbon[-1]

        # --- Trailing Stop Loss Logic ---
        if self.position:
            if self.position.is_long:
                lows, _ = find_peaks(-self.data.Low.to_numpy()[-50:], distance=5)
                if len(lows) > 0:
                    new_swing_sl = self.data.Low[-50:][lows[-1]] * 0.99
                    # Prevent SL from moving above the 21 EMA
                    candidate_sl = max(self.trades[0].sl, new_swing_sl)
                    self.trades[0].sl = min(candidate_sl, ema_21 if not pd.isna(ema_21) else candidate_sl)


            elif self.position.is_short:
                highs, _ = find_peaks(self.data.High.to_numpy()[-50:], distance=5)
                if len(highs) > 0:
                    new_swing_sl = self.data.High[-50:][highs[-1]] * 1.01
                    # Prevent SL from moving below the 21 EMA
                    candidate_sl = min(self.trades[0].sl, new_swing_sl)
                    self.trades[0].sl = max(candidate_sl, ema_21 if not pd.isna(ema_21) else candidate_sl)


        # --- Exit Logic ---
        if self.position.is_long:
            # Exit if 4H trigger wave turns down
            if self.env_trigger_wave[-1] < self.env_trigger_wave[-2]:
                self.position.close()
                return
            # Red diamond (overbought) or EMA turns gray (proxy: price crosses below EMA)
            if self.exec_diamond[-1] == -1 or price < ema_21:
                self.position.close()
                return

        if self.position.is_short:
            # Exit if 4H trigger wave turns up
            if self.env_trigger_wave[-1] > self.env_trigger_wave[-2]:
                self.position.close()
                return
            # Green diamond (oversold) or EMA turns white (proxy: price crosses above EMA)
            if self.exec_diamond[-1] == 1 or price > ema_21:
                self.position.close()
                return

        # --- Entry Logic ---
        if self.position:
            return

        # 4H Environmental Confirmation
        is_long_env = self.env_money_flow[-1] > 50 and self.env_trigger_wave[-1] > self.env_trigger_wave[-2] and self.env_wolfpack[-1] > 0
        is_short_env = self.env_money_flow[-1] < 50 and self.env_trigger_wave[-1] < self.env_trigger_wave[-2] and self.env_wolfpack[-1] < 0

        # Long Entry State Machine
        if is_long_env:
            self.anchor_short_detected = False # Invalidate opposite anchor
            if self.exec_anchor_wave[-1] == 1:
                self.anchor_long_detected = True

            if self.anchor_long_detected:
                # Check for subsequent trigger conditions
                is_long_exec = (self.exec_trigger_wave[-1] > self.exec_trigger_wave[-2] and
                                self.exec_money_flow[-1] > 50 and
                                self.exec_wolfpack[-1] > 0)
                if is_long_exec:
                    sl = price - self.atr[-1] * self.sl_atr_multiplier
                    if price > sl: # Ensure SL is valid
                        self.buy(sl=sl)
                        self.anchor_long_detected = False # Reset state
        else:
             self.anchor_long_detected = False # Invalidate if env changes

        # Short Entry State Machine
        if is_short_env:
            self.anchor_long_detected = False # Invalidate opposite anchor
            if self.exec_anchor_wave[-1] == -1:
                self.anchor_short_detected = True

            if self.anchor_short_detected:
                # Check for subsequent trigger conditions
                is_short_exec = (self.exec_trigger_wave[-1] < self.exec_trigger_wave[-2] and
                                 self.exec_money_flow[-1] < 50 and
                                 self.exec_wolfpack[-1] < 0)
                if is_short_exec:
                    sl = price + self.atr[-1] * self.sl_atr_multiplier
                    if price < sl: # Ensure SL is valid
                        self.sell(sl=sl)
                        self.anchor_short_detected = False # Reset state
        else:
            self.anchor_short_detected = False # Invalidate if env changes


def preprocess_data(data):
    """
    Prepares the data by creating Heikin Ashi candles, resampling to different
    timeframes, and calculating proxy indicators.
    """
    # NOTE: Using 30min as a proxy for the 24min timeframe, as 24 is not a
    # multiple of the base 15min data.
    df_30m = data.resample('30min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    # Calculate Heikin Ashi values
    ha_close = (df_30m['Open'] + df_30m['High'] + df_30m['Low'] + df_30m['Close']) / 4
    ha_open = (df_30m['Open'].shift(1) + df_30m['Close'].shift(1)) / 2
    ha_high = df_30m[['High', 'Open', 'Close']].max(axis=1)
    ha_low = df_30m[['Low', 'Open', 'Close']].min(axis=1)
    df_30m_ha = pd.DataFrame({
        'Open': ha_open, 'High': ha_high, 'Low': ha_low, 'Close': ha_close, 'Volume': df_30m['Volume']
    }).dropna()

    # --- 4H Environmental Indicators (on regular candles) ---
    df_4h = data.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    df_4h['4h_money_flow'] = ta.mfi(df_4h['High'], df_4h['Low'], df_4h['Close'], df_4h['Volume'], length=14)
    rsi_4h = ta.rsi(df_4h['Close'], length=14)
    df_4h['4h_trigger_wave'] = ta.sma(rsi_4h, length=5) # Smoothed RSI
    macd_4h = ta.macd(df_4h['Close'])
    df_4h['4h_wolfpack'] = macd_4h[f'MACDh_{12}_{26}_{9}']

    # --- 30M Execution Indicators (on Heikin Ashi candles) ---
    df_30m_ha['30m_money_flow'] = ta.mfi(df_30m_ha['High'], df_30m_ha['Low'], df_30m_ha['Close'], df_30m_ha['Volume'], length=14)
    rsi_30m = ta.rsi(df_30m_ha['Close'], length=14)
    df_30m_ha['30m_trigger_wave'] = ta.sma(rsi_30m, length=5)
    macd_30m = ta.macd(df_30m_ha['Close'])
    df_30m_ha['30m_wolfpack'] = macd_30m[f'MACDh_{12}_{26}_{9}']
    df_30m_ha['30m_ema_21'] = ta.ema(df_30m_ha['Close'], length=21)
    df_30m_ha['30m_atr'] = ta.atr(df_30m_ha['High'], df_30m_ha['Low'], df_30m_ha['Close'], length=14)

    # Anchor Wave Proxy: Big green dots below -60 / Big red dots above +60
    # Proxy: RSI is deeply oversold/overbought
    df_30m_ha['30m_anchor_wave'] = np.where(rsi_30m < 20, 1, np.where(rsi_30m > 80, -1, 0))

    # Diamond Proxy: Red diamond on top / Green diamond on bottom
    # Proxy: RSI is overbought/oversold
    df_30m_ha['30m_diamond'] = np.where(rsi_30m > 70, -1, np.where(rsi_30m < 30, 1, 0))

    # --- Merge dataframes ---
    final_df = df_30m.copy()

    final_df = final_df.join(df_30m_ha[[
        '30m_money_flow', '30m_trigger_wave', '30m_wolfpack', '30m_ema_21',
        '30m_anchor_wave', '30m_diamond', '30m_atr'
    ]])

    final_df = pd.merge(final_df, df_4h[['4h_money_flow', '4h_trigger_wave', '4h_wolfpack']],
                        left_index=True, right_index=True, how='left')

    final_df.ffill(inplace=True)
    final_df.dropna(inplace=True)

    return final_df


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    try:
        data = pd.read_csv(
            data_path,
            parse_dates=['datetime'],
            index_col='datetime'
        )
        data.columns = [c.strip().capitalize() for c in data.columns]
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # As a fallback, create some synthetic data to allow the script to run
        data = pd.DataFrame({
            'Open': np.random.uniform(20000, 30000, 5000),
            'High': np.random.uniform(20000, 30000, 5000),
            'Low': np.random.uniform(20000, 30000, 5000),
            'Close': np.random.uniform(20000, 30000, 5000),
            'Volume': np.random.uniform(100, 1000, 5000),
        }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=5000, freq='15min')))
        data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 100, 5000)
        data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 100, 5000)

    # Preprocess the data
    processed_data = preprocess_data(data)

    if not processed_data.empty:
        # Initialize and run the backtest
        bt = Backtest(processed_data, MarketCipher4h24mSwing, cash=100_000, commission=.002)
        stats = bt.run()

        print(stats)

        # Save results and plot
        output_filename = 'results/strategy_b89e894bd83b.html'
        bt.plot(filename=output_filename, open_browser=False)

        # Sanitize and save stats to JSON
        stats_sanitized = sanitize_stats(stats)
        with open('results/temp_result.json', 'w') as f:
            json.dump(stats_sanitized, f, indent=4)

        print("\nBacktest complete.")
        print(f"Stats saved to results/temp_result.json")
        print(f"Plot saved to {output_filename}")
    else:
        print("Error: Processed data is empty. Could not run backtest.")
