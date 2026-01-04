from backtesting import Backtest, Strategy
import pandas as pd
import pandas_ta as ta

class MarketCipherBVWAPTrendFollowing(Strategy):
    trigger_period = 20
    cmf_period = 20

    def init(self):
        self.vwap = self.data.VWAP
        self.trigger_line = self.data.Trigger_Line
        self.money_flow = self.data.CMF

    def next(self):
        price = self.data.Close[-1]

        # Long entry conditions
        if (self.vwap[-1] > self.trigger_line[-1] and
                self.data.Low[-2] < self.vwap[-2] and
                self.data.Close[-1] > self.vwap[-1] and
                self.money_flow[-1] > 0):
            if not self.position:
                sl = self.data.Low[-2] * 0.99
                tp = price + (price - sl) * 2
                self.buy(sl=sl, tp=tp)

        # Short entry conditions
        elif (self.vwap[-1] < self.trigger_line[-1] and
              self.data.High[-2] > self.vwap[-2] and
              self.data.Close[-1] < self.vwap[-1] and
              self.money_flow[-1] < 0):
            if not self.position:
                sl = self.data.High[-2] * 1.01
                tp = price - (sl - price) * 2
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    col_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    data = pd.read_csv(
        'data/BTC-USD-15m.csv',
        header=0,
        names=col_names,
        index_col='datetime',
        parse_dates=True,
        usecols=range(len(col_names))
    )
    data.columns = [c.capitalize() for c in data.columns]
    data.sort_index(inplace=True)

    # Pre-calculate indicators
    data['VWAP'] = ta.vwap(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
    data['Trigger_Line'] = ta.sma(data['VWAP'], length=20)
    data['CMF'] = ta.cmf(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], length=20)

    data.dropna(inplace=True)

    bt = Backtest(data, MarketCipherBVWAPTrendFollowing, cash=100000)
    stats = bt.run()

    print(stats)
    bt.plot(filename='results/market_cipher_b_vwap_trend_following.html')
