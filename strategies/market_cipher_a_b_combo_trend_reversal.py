import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Strategy, Backtest
from scipy.signal import find_peaks

# Note: The vumanchu indicators requested in the strategy were not found in the codebase.
# As a result, proxies have been created using standard technical indicators.

def last_swing(series, lookback):
    """Finds the last swing high or low."""
    peaks, _ = find_peaks(series, distance=lookback)
    troughs, _ = find_peaks(-series, distance=lookback)

    if len(peaks) > 0 and len(troughs) > 0:
        if peaks[-1] > troughs[-1]:
            return series[peaks[-1]]
        else:
            return series[troughs[-1]]
    elif len(peaks) > 0:
        return series[peaks[-1]]
    elif len(troughs) > 0:
        return series[troughs[-1]]
    return None

def market_cipher_a(high, low, close, volume):
    """
    Proxy for Market Cipher A.
    Returns VWA (VWAP) and VWB (VWAP moving average).
    """
    vwap = ta.vwap(high, low, close, volume)
    vwb = ta.sma(vwap, length=20)
    return vwap, vwb

def market_cipher_b(high, low, close, volume):
    """
    Proxy for Market Cipher B.
    Returns Money Flow (CMF), Momentum, and MACD.
    """
    money_flow = talib.CMF(high, low, close, volume)
    momentum = talib.MOM(close)
    macd, macdsignal, macdhist = talib.MACD(close)
    return money_flow, momentum, macd, macdsignal, macdhist


class MarketCipherABComboTrendReversal(Strategy):
    # Default parameters
    risk_reward_ratio = 2.0
    swing_lookback = 20
    bounce_lookback = 5

    def init(self):
        # Initialize Market Cipher A indicators
        self.vwa, self.vwb = self.I(
            market_cipher_a,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume
        )

        # Initialize Market Cipher B indicators
        self.money_flow, self.momentum, self.macd, self.macdsignal, self.macdhist = self.I(
            market_cipher_b,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume
        )

    def next(self):
        price = self.data.Close[-1]

        # Long entry conditions
        bounced_off_vwa = any(self.data.Low[-self.bounce_lookback:] <= self.vwa[-self.bounce_lookback:])
        long_ma_condition = price > self.vwa[-1] and price > self.vwb[-1] and self.vwa[-1] > self.vwa[-2] and bounced_off_vwa
        long_money_flow = self.money_flow[-1] > 0
        long_momentum = self.momentum[-1] > 0
        long_macd = self.macd[-1] > self.macdsignal[-1]

        # Short entry conditions
        rejected_vwa = any(self.data.High[-self.bounce_lookback:] >= self.vwa[-self.bounce_lookback:])
        short_ma_condition = price < self.vwa[-1] and price < self.vwb[-1] and self.vwa[-1] < self.vwa[-2] and rejected_vwa
        short_money_flow = self.money_flow[-1] < 0
        short_momentum = self.momentum[-1] < 0
        short_macd = self.macd[-1] < self.macdsignal[-1]

        if self.position.is_long:
            # Long exit conditions
            if self.vwa[-1] < self.vwb[-1] or self.money_flow[-1] < 0 or self.macd[-1] < self.macdsignal[-1] or self.momentum[-1] < 0:
                self.position.close()

        elif self.position.is_short:
            # Short exit conditions
            if self.vwa[-1] > self.vwb[-1] or self.money_flow[-1] > 0 or self.macd[-1] > self.macdsignal[-1] or self.momentum[-1] > 0:
                self.position.close()

        if not self.position:
            # Long entry
            if long_ma_condition and long_money_flow and long_momentum and long_macd:
                last_low = last_swing(self.data.Low, self.swing_lookback)
                if last_low and last_low < price:
                    sl = last_low
                    tp = price + (price - sl) * self.risk_reward_ratio
                    self.buy(sl=sl, tp=tp)

            # Short entry
            elif short_ma_condition and short_money_flow and short_momentum and short_macd:
                last_high = last_swing(self.data.High, self.swing_lookback)
                if last_high and last_high > price:
                    sl = last_high
                    tp = price - (sl - price) * self.risk_reward_ratio
                    self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)

    # Initialize and run backtest
    bt = Backtest(data, MarketCipherABComboTrendReversal, cash=100000, commission=.002)
    stats = bt.run()

    # Print stats and save results
    print(stats)
    bt.plot(filename='results/market_cipher_a_b_combo_trend_reversal.html')

    # Save results to JSON
    stats_df = pd.DataFrame(stats).transpose()
    stats_df.to_json("results/temp_result.json")
