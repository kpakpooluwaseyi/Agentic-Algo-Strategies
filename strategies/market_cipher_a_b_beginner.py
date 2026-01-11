
from backtesting import Strategy
from src.indicators.vumanchu import cipher_b
import pandas_ta as ta
import pandas as pd

def preprocess_data(df: pd.DataFrame, atr_period=14, volume_ma_period=20, trend_ema_period=200):
    """
    Applies the necessary indicators for the Market Cipher Beginner strategy.
    """
    df = df.copy()

    # Apply Cipher B indicators
    df = cipher_b(df)

    # Convert boolean signals for backtesting.py compatibility
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # ATR for risk management
    df.ta.atr(length=atr_period, append=True)

    # Volume MA for confirmation
    df['volume_ma'] = df['Volume'].rolling(window=volume_ma_period).mean()

    # 4H Trend Filter
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).copy()

    df_4h['ema_trend'] = df_4h.ta.ema(length=trend_ema_period)

    # Merge 4H EMA back to 15m DataFrame
    df = pd.merge(df, df_4h[['ema_trend']], left_index=True, right_index=True, how='left')
    df['ema_trend'] = df['ema_trend'].ffill()

    return df

class MarketCipherBeginner(Strategy):
    # Strategy parameters
    atr_period = 14
    volume_ma_period = 20
    trend_ema_period = 200
    atr_multiplier_sl = 2
    atr_multiplier_tp = 3

    def init(self):
        # Indicators
        self.wt_buy_signal = self.I(lambda: self.data.buy_signal, name="wt_buy_signal")
        self.wt_sell_signal = self.I(lambda: self.data.sell_signal, name="wt_sell_signal")
        self.money_flow = self.I(lambda: self.data.rsimfi, name="money_flow")
        self.stoch_k = self.I(lambda: self.data.stoch_rsi_k, name="stoch_k")
        self.stoch_d = self.I(lambda: self.data.stoch_rsi_d, name="stoch_d")
        self.momentum_waves = self.I(lambda: self.data.wt_vwap, name="momentum_waves")

        # ATR indicator with dynamic name
        atr_col_name = f"ATR_{self.atr_period}"
        self.atr = self.I(lambda: self.data.df[atr_col_name], name="atr")

        self.volume_ma = self.I(lambda: self.data.volume_ma, name="volume_ma")
        self.ema_trend = self.I(lambda: self.data.ema_trend, name="ema_trend")

    def next(self):
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]

        # Exit logic for existing trades
        if self.position:
            if self.position.is_long and self.wt_sell_signal[-1]:
                self.position.close()
            elif self.position.is_short and self.wt_buy_signal[-1]:
                self.position.close()

        # Entry logic
        if not self.position:
            # Long entry conditions
            long_condition = (
                self.wt_buy_signal[-1] == 1 and
                self.money_flow[-1] > 0 and
                self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-2] <= self.stoch_d[-2] and self.stoch_k[-1] < 20 and
                self.momentum_waves[-1] > 0 and
                volume > self.volume_ma[-1] and
                price > self.ema_trend[-1]
            )

            if long_condition:
                sl = price - self.atr[-1] * self.atr_multiplier_sl
                tp = price + self.atr[-1] * self.atr_multiplier_tp
                self.buy(sl=sl, tp=tp)

            # Short entry conditions
            short_condition = (
                self.wt_sell_signal[-1] == 1 and
                self.money_flow[-1] < 0 and
                self.stoch_k[-1] < self.stoch_d[-1] and self.stoch_k[-2] >= self.stoch_d[-2] and self.stoch_k[-1] > 80 and
                self.momentum_waves[-1] < 0 and
                volume > self.volume_ma[-1] and
                price < self.ema_trend[-1]
            )

            if short_condition:
                sl = price + self.atr[-1] * self.atr_multiplier_sl
                tp = price - self.atr[-1] * self.atr_multiplier_tp
                self.sell(sl=sl, tp=tp)
