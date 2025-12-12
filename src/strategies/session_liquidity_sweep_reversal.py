from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
import json

class SessionLiquiditySweepReversalStrategy(Strategy):
    asia_start_hour = 0
    asia_end_hour = 8
    london_start_hour = 9
    asia_range_percent_filter = 2.0
    trailing_sl_pips = 50

    def init(self):
        self.asia_high = self.data.AsiaHigh
        self.asia_low = self.data.AsiaLow
        self.asia_range = self.asia_high - self.asia_low
        self.asia_range_percent = (self.asia_range / self.asia_low) * 100

    def next(self):
        current_time = self.data.index[-1]

        # --- Filters ---
        if current_time.hour < self.london_start_hour:
            return

        if self.asia_range_percent[-1] > self.asia_range_percent_filter:
            return

        # --- Trailing Stop Logic ---
        for trade in self.trades:
            if trade.is_long:
                new_sl = self.data.Close[-1] - (self.trailing_sl_pips * 0.0001)
                if new_sl > trade.sl:
                    trade.sl = new_sl
            else: # Short trade
                new_sl = self.data.Close[-1] + (self.trailing_sl_pips * 0.0001)
                if new_sl < trade.sl:
                    trade.sl = new_sl

        # --- Short Entry ---
        is_bearish_reversal = self.is_bearish_engulfing(self.data.Open[-1], self.data.Close[-1], self.data.Open[-2], self.data.Close[-2]) or \
                              self.is_inverted_hammer(self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1])

        if self.data.High[-1] > self.asia_high[-1] and is_bearish_reversal and self.data.Close[-1] < self.asia_high[-1]:
            sl = self.data.High[-1]
            self.sell(sl=sl, tp=self.asia_low[-1], size=0.5)
            self.sell(sl=sl, size=0.5)

        # --- Long Entry ---
        is_bullish_reversal = self.is_bullish_engulfing(self.data.Open[-1], self.data.Close[-1], self.data.Open[-2], self.data.Close[-2]) or \
                              self.is_hammer(self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1])

        if self.data.Low[-1] < self.asia_low[-1] and is_bullish_reversal and self.data.Close[-1] > self.asia_low[-1]:
            sl = self.data.Low[-1]
            self.buy(sl=sl, tp=self.asia_high[-1], size=0.5)
            self.buy(sl=sl, size=0.5)

    def is_bearish_engulfing(self, open1, close1, open2, close2):
        # Current candle is bearish, previous is bullish and engulfed
        return open1 > close1 and open2 < close2 and open1 >= close2 and close1 <= open2

    def is_bullish_engulfing(self, open1, close1, open2, close2):
        # Current candle is bullish, previous is bearish and engulfed
        return open1 < close1 and open2 > close2 and open1 <= close2 and close1 >= open2

    def is_hammer(self, open, high, low, close):
        body = abs(close - open)
        lower_wick = open - low if open < close else close - low
        upper_wick = high - close if open < close else high - open
        return lower_wick > 2 * body and upper_wick < body

    def is_inverted_hammer(self, open, high, low, close):
        body = abs(close - open)
        lower_wick = open - low if open < close else close - low
        upper_wick = high - close if open < close else high - open
        return upper_wick > 2 * body and lower_wick < body

if __name__ == '__main__':
    # Generate synthetic 15M data for 90 days
    time_range = pd.date_range('2023-01-01', '2023-03-31', freq='15min', tz='UTC')
    price = 1.10
    volatility = 0.0002
    prices = []
    for _ in time_range:
        price += np.random.normal(0, volatility)
        prices.append(price)

    data = pd.DataFrame({'Open': prices, 'High': prices, 'Low': prices, 'Close': prices}, index=time_range)
    data['Open'] = data['Close'].shift(1)
    data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, volatility, len(data))
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, volatility, len(data))
    data = data.dropna()

    # Pre-compute Asia session H/L
    asia_start_hour = 0
    asia_end_hour = 8
    df = data

    asia_session_mask = (df.index.hour >= asia_start_hour) & (df.index.hour < asia_end_hour)
    daily_asia_stats = df[asia_session_mask].groupby(df[asia_session_mask].index.date).agg(
        AsiaHigh=('High', 'max'),
        AsiaLow=('Low', 'min')
    )
    daily_asia_stats = daily_asia_stats.shift(1)

    df['AsiaHigh'] = df.index.date
    df['AsiaHigh'] = df['AsiaHigh'].map(daily_asia_stats['AsiaHigh'])
    df['AsiaLow'] = df.index.date
    df['AsiaLow'] = df['AsiaLow'].map(daily_asia_stats['AsiaLow'])

    df['AsiaHigh'] = df['AsiaHigh'].ffill()
    df['AsiaLow'] = df['AsiaLow'].ffill()
    df = df.dropna()

    bt = Backtest(df, SessionLiquiditySweepReversalStrategy, cash=10000, commission=.002, trade_on_close=True, exclusive_orders=True, finalize_trades=True)


    # Optimize
    stats = bt.optimize(
        asia_range_percent_filter=list(np.arange(1.0, 4.0, 0.5)),
        trailing_sl_pips=[20, 30, 50, 70],
        maximize='Return [%]'
    )

    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'session_liquidity_sweep_reversal',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    # Generate plot
    bt.plot()
