"""
Market Cipher Triple Confirmation Strategy
=========================================
A trend-following and momentum strategy based on the confluence of signals
from the Market Cipher B and A indicators, aligned with a higher-timeframe trend.

Long Entry Conditions:
1. Market Cipher B Green Dot (WaveTrend cross up in oversold).
2. Money Flow is positive and rising.
3. Momentum Waves (WaveTrend VWAP) are positive.
4. Stochastic RSI is crossing up from the oversold region.
5. Higher-timeframe trend (e.g., 4H EMA) is bullish.
6. Volume is above its moving average.

Short Entry Conditions:
1. Market Cipher B Red Dot (WaveTrend cross down in overbought).
2. Money Flow is negative and falling.
3. Momentum Waves (WaveTrend VWAP) are negative.
4. Stochastic RSI is crossing down from the overbought region.
5. Higher-timeframe trend (e.g., 4H EMA) is bearish.
6. Volume is above its moving average.

Exit Conditions:
- Primary: Opposite Market Cipher B dot appears.
- Risk Management: ATR-based Stop Loss (2x) and Take Profit (3x).
"""

from backtesting import Strategy, Backtest
from backtesting.lib import crossover
import pandas as pd
import talib
from src.indicators.vumanchu import cipher_b
from scipy.signal import find_peaks


def preprocess_data(df: pd.DataFrame, htf_ema_period=200, volume_ma_period=20, atr_period=14, sr_lookback=20) -> pd.DataFrame:
    """
    Applies all necessary indicators and filters to the raw OHLCV data.
    """
    df = df.copy()

    # 1. Add VuManchu Cipher B Indicators
    df = cipher_b(df)

    # 2. Add Higher-Timeframe (4H) Trend Filter
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema'] = talib.EMA(df_4h['Close'], timeperiod=htf_ema_period)
    df_4h['htf_trend_up'] = df_4h['Close'] > df_4h['ema']

    # Map 4H trend back to the original timeframe
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')
    df['htf_trend_up'].fillna(False, inplace=True) # Fill initial NaNs

    # 3. Add Volume Confirmation
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=volume_ma_period)

    # 4. Add ATR for Risk Management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_period)

    # 5. Add Support and Resistance Levels using find_peaks
    resistance_indices, _ = find_peaks(df['High'], distance=sr_lookback)
    support_indices, _ = find_peaks(-df['Low'], distance=sr_lookback)

    df['resistance'] = pd.NA
    df['support'] = pd.NA

    df.loc[df.index[resistance_indices], 'resistance'] = df['High'].iloc[resistance_indices]
    df.loc[df.index[support_indices], 'support'] = df['Low'].iloc[support_indices]

    df['resistance'] = df['resistance'].ffill()
    df['support'] = df['support'].ffill()

    return df


class MarketCipherTripleConfirmation(Strategy):
    # --- Optimizable Parameters ---

    # ATR Risk Management
    atr_period = 14
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    # Volume Confirmation
    volume_ma_period = 20

    # Higher Timeframe Trend Filter
    htf_ema_period = 200 # Using 200 EMA on the 4H chart

    # Support/Resistance
    sr_lookback = 20 # How many bars to look back for a swing high/low
    sr_proximity_pct = 0.01 # How close price must be to S/R (1%)

    # Stochastic RSI Parameters
    stoch_rsi_oversold = 20
    stoch_rsi_overbought = 80

    def init(self):
        """
        Initialize indicators. The data is preprocessed, so we just need to
        create indicator wrappers for easy access in the `next` method.
        """
        # --- Create indicator wrappers ---

        # Market Cipher B Signals
        self.buy_signal = self.I(lambda: self.data.buy_signal.astype(float))
        self.sell_signal = self.I(lambda: self.data.sell_signal.astype(float))

        # Money Flow
        self.money_flow = self.I(lambda: self.data.rsimfi)

        # Momentum Waves (using wt_vwap as per Cipher A logic)
        self.momentum_wave = self.I(lambda: self.data.wt_vwap)

        # Stochastic RSI
        self.stoch_k = self.I(lambda: self.data.stoch_rsi_k)
        self.stoch_d = self.I(lambda: self.data.stoch_rsi_d)

        # Higher-Timeframe Trend
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up.astype(float))

        # Volume Confirmation
        self.volume_ma = self.I(lambda: self.data.volume_ma)

        # ATR for Risk Management
        self.atr = self.I(lambda: self.data.atr)

        # Support / Resistance
        self.support = self.I(lambda: self.data.support)
        self.resistance = self.I(lambda: self.data.resistance)

    def next(self):
        """
        Defines the trading logic for each bar.
        """
        price = self.data.Close[-1]

        # --- FILTERS: Conditions that must be met to consider any trade ---
        volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]

        # --- ENTRY CONDITIONS ---

        # Stoch RSI bullish cross
        stoch_bullish_cross = crossover(self.stoch_k, self.stoch_d) and \
                              self.stoch_k[-1] < self.stoch_rsi_oversold

        # Stoch RSI bearish cross
        stoch_bearish_cross = crossover(self.stoch_d, self.stoch_k) and \
                               self.stoch_k[-1] > self.stoch_rsi_overbought

        # Long entry: All bullish signals align
        long_conditions_met = (
            self.buy_signal[-1] == 1 and
            self.money_flow[-1] > 0 and
            self.momentum_wave[-1] > 0 and
            stoch_bullish_cross and
            self.htf_trend_up[-1] == 1 and
            volume_confirmed
        )

        # Short entry: All bearish signals align
        short_conditions_met = (
            self.sell_signal[-1] == 1 and
            self.money_flow[-1] < 0 and
            self.momentum_wave[-1] < 0 and
            stoch_bearish_cross and
            self.htf_trend_up[-1] == 0 and
            volume_confirmed
        )

        # --- TRADE EXECUTION ---

        # If no position is open, check for new entry signals
        if not self.position:
            if long_conditions_met:
                # Calculate SL and TP
                sl = price - self.atr[-1] * self.atr_sl_multiplier
                tp = price + self.atr[-1] * self.atr_tp_multiplier
                self.buy(sl=sl, tp=tp)

            elif short_conditions_met:
                # Calculate SL and TP
                sl = price + self.atr[-1] * self.atr_sl_multiplier
                tp = price - self.atr[-1] * self.atr_tp_multiplier
                self.sell(sl=sl, tp=tp)

        # --- EXIT CONDITIONS for open positions ---
        else:
            if self.position.is_long and self.sell_signal[-1] == 1:
                self.position.close()
            elif self.position.is_short and self.buy_signal[-1] == 1:
                self.position.close()


if __name__ == '__main__':
    # --- Backtest Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    cash = 100_000
    commission = 0.002

    # Instantiate the strategy to access its parameters
    strategy = MarketCipherTripleConfirmation

    # --- Data Loading and Preprocessing ---
    try:
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # As a fallback, create some synthetic data for testing
        print("Generating synthetic data for demonstration...")
        from backtesting.test import EURUSD
        df = EURUSD.copy() # Using built-in data as a fallback

    # Apply the preprocessing function using parameters from the strategy class
    df_processed = preprocess_data(
        df,
        htf_ema_period=strategy.htf_ema_period,
        volume_ma_period=strategy.volume_ma_period,
        atr_period=strategy.atr_period,
        sr_lookback=strategy.sr_lookback
    )

    # Remove rows with NaN values resulting from indicator calculations
    df_processed.dropna(inplace=True)

    # --- Run Backtest ---
    bt = Backtest(df_processed, MarketCipherTripleConfirmation, cash=cash, commission=commission)
    stats = bt.run()

    print("\n--- Backtest Results ---")
    print(stats)

    # --- Save plot and results ---
    import os
    import json

    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    plot_filename = 'results/market_cipher_triple_confirmation.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"\nBacktest plot saved to {plot_filename}")

    # Save stats to a temporary JSON file
    stats_dict = dict(stats)
    # Convert non-serializable items
    if '_strategy' in stats_dict:
        del stats_dict['_strategy']
    if '_equity_curve' in stats_dict:
        del stats_dict['_equity_curve']
    if '_trades' in stats_dict:
        del stats_dict['_trades']

    results_filename = 'results/temp_result.json'
    with open(results_filename, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Backtest stats saved to {results_filename}")
