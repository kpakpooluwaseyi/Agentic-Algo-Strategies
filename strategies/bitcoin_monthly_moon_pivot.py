from backtesting import Strategy
import pandas as pd

# --- Hardcoded Moon Phase Data (2020-2024) ---
# Source: Griffith Observatory, NASA
NEW_MOON_DATES = [
    # 2020
    "2020-01-24", "2020-02-23", "2020-03-24", "2020-04-22", "2020-05-22",
    "2020-06-21", "2020-07-20", "2020-08-18", "2020-09-17", "2020-10-16",
    "2020-11-15", "2020-12-14",
    # 2021
    "2021-01-13", "2021-02-11", "2021-03-13", "2021-04-11", "2021-05-11",
    "2021-06-10", "2021-07-09", "2021-08-08", "2021-09-06", "2021-10-06",
    "2021-11-04", "2021-12-04",
    # 2022
    "2022-01-02", "2022-01-31", "2022-03-02", "2022-03-31", "2022-04-30",
    "2022-05-30", "2022-06-28", "2022-07-28", "2022-08-27", "2022-09-25",
    "2022-10-25", "2022-11-23", "2022-12-23",
    # 2023
    "2023-01-21", "2023-02-19", "2023-03-21", "2023-04-19", "2023-05-19",
    "2023-06-17", "2023-07-17", "2023-08-16", "2023-09-14", "2023-10-14",
    "2023-11-13", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-09", "2024-03-10", "2024-04-08", "2024-05-07",
    "2024-06-06", "2024-07-05", "2024-08-04", "2024-09-02", "2024-10-02",
    "2024-11-01", "2024-11-30", "2024-12-30"
]

FULL_MOON_DATES = [
    # 2020
    "2020-01-10", "2020-02-09", "2020-03-09", "2020-04-07", "2020-05-07",
    "2020-06-05", "2020-07-05", "2020-08-03", "2020-09-02", "2020-10-01",
    "2020-10-31", "2020-11-30", "2020-12-29",
    # 2021
    "2021-01-28", "2021-02-27", "2021-03-28", "2021-04-26", "2021-05-26",
    "2021-06-24", "2021-07-23", "2021-08-22", "2021-09-20", "2021-10-20",
    "2021-11-19", "2021-12-18",
    # 2022
    "2022-01-17", "2022-02-16", "2022-03-18", "2022-04-16", "2022-05-15",
    "2022-06-14", "2022-07-13", "2022-08-11", "2022-09-10", "2022-10-09",
    "2022-11-08", "2022-12-07",
    # 2023
    "2023-01-06", "2023-02-05", "2023-03-07", "2023-04-05", "2023-05-05",
    "2023-06-03", "2023-07-03", "2023-08-01", "2023-08-30", "2023-09-29",
    "2023-10-28", "2023-11-27", "2023-12-26",
    # 2024
    "2024-01-25", "2024-02-24", "2024-03-25", "2024-04-23", "2024-05-23",
    "2024-06-21", "2024-07-21", "2024-08-19", "2024-09-17", "2024-10-17",
    "2024-11-15", "2024-12-15"
]

class BitcoinMonthlyMoonPivotStrategy(Strategy):
    rr = 3  # Risk/Reward Ratio

    def init(self):
        self.new_moons = {pd.to_datetime(d).date() for d in NEW_MOON_DATES}
        self.full_moons = {pd.to_datetime(d).date() for d in FULL_MOON_DATES}
        self.current_month = -1
        self.trade_active_this_month = False
        self.pivot_high = None
        self.pivot_low = None
        self.pivot_high_candle_low = None
        self.pivot_low_candle_high = None
        self.seeking_short_entry = False
        self.seeking_long_entry = False

    def is_near_moon_event(self, current_date, moon_dates, window=3):
        for moon_date in moon_dates:
            if abs((current_date - moon_date).days) <= window:
                return True
        return False

    def next(self):
        current_date = self.data.index[-1].date()
        current_day = current_date.day
        current_month = current_date.month

        if current_month != self.current_month:
            self.current_month = current_month
            self.trade_active_this_month = False
            self.pivot_high = None
            self.pivot_low = None
            self.seeking_short_entry = False
            self.seeking_long_entry = False

        if self.position and self.data.index[-1].is_month_end:
            self.position.close()
            return

        if not self.position and not self.trade_active_this_month and 1 <= current_day <= 12:
            is_near_new_moon = self.is_near_moon_event(current_date, self.new_moons)
            is_near_full_moon = self.is_near_moon_event(current_date, self.full_moons)

            if len(self.data.High) < 3:
                return

            if is_near_new_moon and not self.seeking_long_entry and self.pivot_high is None:
                if self.data.High[-2] > self.data.High[-3] and self.data.High[-2] > self.data.High[-1]:
                    self.pivot_high = self.data.High[-2]
                    self.pivot_high_candle_low = self.data.Low[-2]
                    self.seeking_short_entry = True

            if is_near_full_moon and not self.seeking_short_entry and self.pivot_low is None:
                if self.data.Low[-2] < self.data.Low[-3] and self.data.Low[-2] < self.data.Low[-1]:
                    self.pivot_low = self.data.Low[-2]
                    self.pivot_low_candle_high = self.data.High[-2]
                    self.seeking_long_entry = True

            if self.seeking_short_entry and self.data.Close[-1] < self.pivot_high_candle_low:
                sl = self.pivot_high
                risk = sl - self.data.Close[-1]
                if risk > 0:
                    tp = self.data.Close[-1] - risk * self.rr
                    size = int((self.equity * 0.01) / risk)
                    if size > 0:
                        self.sell(sl=sl, tp=tp, size=size)
                    self.trade_active_this_month = True
                    self.seeking_short_entry = False

            if self.seeking_long_entry and self.data.Close[-1] > self.pivot_low_candle_high:
                sl = self.pivot_low
                risk = self.data.Close[-1] - sl
                if risk > 0:
                    tp = self.data.Close[-1] + risk * self.rr
                    size = int((self.equity * 0.01) / risk)
                    if size > 0:
                        self.buy(sl=sl, tp=tp, size=size)
                    self.trade_active_this_month = True
                    self.seeking_long_entry = False


if __name__ == '__main__':
    from backtesting import Backtest
    import pandas as pd
    import numpy as np
    import json
    import os

    # --- Generate Synthetic Data ---
    def generate_synthetic_data(start_date='2020-01-01', end_date='2024-12-31'):
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        price = 10000 + np.random.randn(len(dates)).cumsum() * 100
        data = pd.DataFrame({
            'Open': price,
            'High': price + np.abs(np.random.randn(len(dates)) * 50),
            'Low': price - np.abs(np.random.randn(len(dates)) * 50),
            'Close': price + np.random.randn(len(dates)) * 20,
            'Volume': np.random.randint(100, 1000, len(dates))
        }, index=dates)
        return data

    data = generate_synthetic_data()

    # --- Run Backtest and Optimization ---
    bt = Backtest(data, BitcoinMonthlyMoonPivotStrategy, cash=1_000_000, commission=.002)

    stats = bt.optimize(rr=range(2, 5, 1),
                        maximize='Sharpe Ratio')

    # --- Output Results ---
    print(stats)
    os.makedirs('results', exist_ok=True)

    # Handle cases where no trades were made
    win_rate = stats['Win Rate [%]'] if stats['# Trades'] > 0 else 0
    sharpe = stats['Sharpe Ratio'] if stats['# Trades'] > 0 else 0

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'bitcoin_monthly_moon_pivot',
            'return': stats['Return [%]'],
            'sharpe': sharpe,
            'max_drawdown': stats['Max. Drawdown [%]'],
            'win_rate': win_rate,
            'total_trades': stats['# Trades']
        }, f, indent=2)

    bt.plot(filename='results/moon_pivot_plot.html')
