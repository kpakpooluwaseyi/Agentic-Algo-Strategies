import pandas as pd
import talib
from src.indicators.vumanchu import cipher_b
from src.strategies.base import MoonDevStrategy


def preprocess_data(df: pd.DataFrame, htf_ema_period=200, volume_ma_period=20, atr_period=14) -> pd.DataFrame:
    """
    Adds the required indicators for the MarketCipherBMoneyFlowDots strategy.
    - VuManchu Cipher B indicators
    - Higher timeframe (4H) trend filter
    - ATR for risk management
    - Volume MA for confirmation
    """
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # Higher Timeframe Trend Filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema'] = talib.EMA(df_4h['Close'], timeperiod=htf_ema_period)
    df_4h['htf_trend_up'] = df_4h['Close'] > df_4h['ema']
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')

    # ATR for Risk Management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_period)

    # Volume Confirmation
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=volume_ma_period)

    return df


class MarketCipherBMoneyFlowDots(MoonDevStrategy):
    """
    Strategy Name: market_cipher_b_money_flow_dots
    Strategy Type: Momentum, Trend-following
    Timeframe: 5m, 15m, 1h
    Instruments: Crypto, Forex

    Entry Rules:
    Long Entry:
    - Market Cipher B (Money Flow): Money flow crosses above the zero line and turns green.
    - Market Cipher B (Waves): Green dot appears.
    - Confirmation: Price action confirming bullish move.

    Short Entry:
    - Market Cipher B (Money Flow): Money flow crosses below the zero line and turns red.
    - Market Cipher B (Waves): Red dot appears.
    - Confirmation: Price action confirming bearish move.
    """
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_period = 20
    htf_ema_period = 200

    def init(self):
        # For backtesting.py to correctly plot indicators, they must be initialized here
        self.buy_sig = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_sig = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.money_flow = self.I(lambda: self.data.rsimfi, name='money_flow')
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name='htf_trend_up')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')

    def next(self):
        # Wait for indicator warmup
        if len(self.data.Close) < self.htf_ema_period:
            return

        # --- Filters ---
        # 1. Higher Timeframe Trend Filter
        in_uptrend = self.htf_trend_up[-1]

        # 2. Volume Confirmation
        volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]
        if not volume_confirmed:
            return

        # --- Entry Logic ---
        if not self.position:
            # Long Entry
            if self.buy_sig[-1] and self.money_flow[-1] > 0 and in_uptrend:
                sl = self.data.Close[-1] - (self.atr[-1] * self.atr_sl_multiplier)
                tp = self.data.Close[-1] + (self.atr[-1] * self.atr_tp_multiplier)
                self.buy(sl=sl, tp=tp)

            # Short Entry
            elif self.sell_sig[-1] and self.money_flow[-1] < 0 and not in_uptrend:
                sl = self.data.Close[-1] + (self.atr[-1] * self.atr_sl_multiplier)
                tp = self.data.Close[-1] - (self.atr[-1] * self.atr_tp_multiplier)
                self.sell(sl=sl, tp=tp)

        # --- Exit Logic ---
        else:
            if self.position.is_long and (self.sell_sig[-1] or self.money_flow[-1] < 0):
                self.position.close()
            elif self.position.is_short and (self.buy_sig[-1] or self.money_flow[-1] > 0):
                self.position.close()


if __name__ == '__main__':
    from backtesting import Backtest
    import os
    import sys

    # Add the project root to the Python path
    # This allows for the correct import of the 'src' module
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.strategies.base import MoonDevStrategy

    # Set the data path
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/BTC-USD-15m.csv')

    # Load the data
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    # Preprocess the data
    df = preprocess_data(df)
    df = df.dropna()

    # Initialize the backtest
    bt = Backtest(df, MarketCipherBMoneyFlowDots, cash=100_000, commission=.002)

    # Run the backtest
    stats = bt.run()

    # Print the results
    print(stats)

    # Save the plot
    plot_filename = 'results/market_cipher_b_money_flow_dots.html'
    bt.plot(filename=plot_filename)

    # Save the stats
    stats_df = pd.DataFrame([stats]).drop(columns=['_trades', '_equity_curve', '_strategy'], errors='ignore')
    stats_df.to_json("results/temp_result.json")
