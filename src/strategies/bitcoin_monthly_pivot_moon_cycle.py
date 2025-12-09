from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
import json

# ATR Indicator helper function
def ATR(high, low, close, n):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return atr.to_numpy()

class BitcoinMonthlyPivotMoonCycleStrategy(Strategy):
    # Strategy parameters
    atr_period = 14
    atr_multiplier = 1.5

    def init(self):
        # Initialize indicators
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        # Get the current date
        current_date = self.data.index[-1]
        day_of_month = current_date.day

        # Time-based exit
        if day_of_month == 28 and self.position:
            self.position.close()

        # Entry rules
        if 1 <= day_of_month <= 12 and not self.position:
            # Long entry
            if self.is_full_moon_window(current_date) and self.is_bullish_reversal():
                sl = self.data.Low[-1] - self.atr[-1] * self.atr_multiplier
                size = self.calculate_size(sl, trade_is_long=True)
                if size > 0:
                    self.buy(sl=sl, size=size)

            # Short entry
            elif self.is_new_moon_window(current_date) and self.is_bearish_reversal():
                sl = self.data.High[-1] + self.atr[-1] * self.atr_multiplier
                size = self.calculate_size(sl, trade_is_long=False)
                if size > 0:
                    self.sell(sl=sl, size=size)

    def calculate_size(self, sl, trade_is_long):
        """Calculates position size based on 1% risk."""
        risk_per_trade = 0.01 * self.equity
        price = self.data.Close[-1]
        risk_per_unit = price - sl if trade_is_long else sl - price

        if risk_per_unit <= 0:
            return 0

        size_in_units = risk_per_trade / risk_per_unit

        # Since fractional trading is not supported, return integer number of units
        return int(size_in_units)

    def is_full_moon_window(self, current_date):
        # A simple implementation with a hardcoded list of full moon dates
        # In a real scenario, this would be calculated or fetched from a reliable source
        full_moon_dates = pd.to_datetime([
            '2020-01-10', '2020-02-09', '2020-03-09', '2020-04-08', '2020-05-07', '2020-06-05',
            '2020-07-05', '2020-08-03', '2020-09-02', '2020-10-01', '2020-10-31', '2020-11-30',
            '2020-12-30', '2021-01-28', '2021-02-27', '2021-03-28', '2021-04-27', '2021-05-26',
            '2021-06-24', '2021-07-24', '2021-08-22', '2021-09-20', '2021-10-20', '2021-11-19',
            '2021-12-19', '2022-01-18', '2022-02-16', '2022-03-18', '2022-04-16', '2022-05-16',
            '2022-06-14', '2022-07-13', '2022-08-12', '2022-09-10', '2022-10-09', '2022-11-08',
            '2022-12-08', '2023-01-06', '2023-02-05', '2023-03-07', '2023-04-06', '2023-05-05',
            '2023-06-04', '2023-07-03', '2023-08-01', '2023-08-31', '2023-09-29', '2023-10-28',
            '2023-11-27', '2023-12-27'
        ])
        return any(abs(current_date - fm_date).days <= 3 for fm_date in full_moon_dates)

    def is_new_moon_window(self, current_date):
        # A simple implementation with a hardcoded list of new moon dates
        new_moon_dates = pd.to_datetime([
            '2020-01-24', '2020-02-23', '2020-03-24', '2020-04-23', '2020-05-22', '2020-06-21',
            '2020-07-20', '2020-08-19', '2020-09-17', '2020-10-16', '2020-11-15', '2020-12-14',
            '2021-01-13', '2021-02-11', '2021-03-13', '2021-04-12', '2021-05-11', '2021-06-10',
            '2021-07-10', '2021-08-08', '2021-09-07', '2021-10-06', '2021-11-04', '2021-12-04',
            '2022-01-02', '2022-02-01', '2022-03-02', '2022-04-01', '2022-04-30', '2022-05-30',
            '2022-06-29', '2022-07-28', '2022-08-27', '2022-09-25', '2022-10-25', '2022-11-23',
            '2022-12-23', '2023-01-21', '2023-02-20', '2023-03-21', '2023-04-20', '2023-05-19',
            '2023-06-18', '2023-07-17', '2023-08-16', '2023-09-15', '2023-10-14', '2023-11-13',
            '2023-12-12'
        ])
        return any(abs(current_date - nm_date).days <= 3 for nm_date in new_moon_dates)

    def is_bullish_reversal(self):
        # Check for Hammer
        if self.is_hammer():
            return True
        # Check for Bullish Engulfing
        if self.is_bullish_engulfing():
            return True
        return False

    def is_bearish_reversal(self):
        # Check for Shooting Star
        if self.is_shooting_star():
            return True
        # Check for Bearish Engulfing
        if self.is_bearish_engulfing():
            return True
        return False

    def is_hammer(self):
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        lower_shadow = self.data.Open[-1] - self.data.Low[-1] if self.data.Close[-1] > self.data.Open[-1] else self.data.Close[-1] - self.data.Low[-1]
        upper_shadow = self.data.High[-1] - self.data.Close[-1] if self.data.Close[-1] > self.data.Open[-1] else self.data.High[-1] - self.data.Open[-1]
        return body > 0 and lower_shadow >= 2 * body and upper_shadow < body

    def is_bullish_engulfing(self):
        return (self.data.Close[-2] < self.data.Open[-2] and
                self.data.Close[-1] > self.data.Open[-1] and
                self.data.Close[-1] >= self.data.Open[-2] and
                self.data.Open[-1] <= self.data.Close[-2])

    def is_shooting_star(self):
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        upper_shadow = self.data.High[-1] - self.data.Close[-1] if self.data.Close[-1] > self.data.Open[-1] else self.data.High[-1] - self.data.Open[-1]
        lower_shadow = self.data.Open[-1] - self.data.Low[-1] if self.data.Close[-1] > self.data.Open[-1] else self.data.Close[-1] - self.data.Low[-1]
        return body > 0 and upper_shadow >= 2 * body and lower_shadow < body

    def is_bearish_engulfing(self):
        return (self.data.Close[-2] > self.data.Open[-2] and
                self.data.Close[-1] < self.data.Open[-1] and
                self.data.Open[-1] >= self.data.Close[-2] and
                self.data.Close[-1] <= self.data.Open[-2])

def generate_synthetic_data(start_date='2020-01-01', end_date='2023-12-31', initial_price=10000):
    """Generates synthetic daily OHLC data."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    price = np.zeros(n)
    price[0] = initial_price

    # Generate a random walk for the close price
    returns = np.random.randn(n) * 0.02  # Daily volatility of 2%
    price = initial_price * np.exp(np.cumsum(returns))

    # Create OHLC data
    open_price = price * (1 + np.random.uniform(-0.01, 0.01, n))
    close_price = price
    high_price = np.maximum(open_price, close_price) * (1 + np.random.uniform(0, 0.01, n))
    low_price = np.minimum(open_price, close_price) * (1 - np.random.uniform(0, 0.01, n))

    data = pd.DataFrame({
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price
    }, index=dates)

    return data

if __name__ == '__main__':
    # Load or generate data
    data = generate_synthetic_data()

    # Run backtest
    bt = Backtest(data, BitcoinMonthlyPivotMoonCycleStrategy, cash=1_000_000, commission=.002)

    # Optimize
    stats = bt.optimize(atr_period=range(10, 20, 2),
                         atr_multiplier=list(np.arange(1.5, 3.0, 0.5)),
                         maximize='Sharpe Ratio')

    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'bitcoin_monthly_pivot_moon_cycle',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    # Generate plot
    bt.plot()
