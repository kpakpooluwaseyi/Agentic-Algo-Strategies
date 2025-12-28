import json
import os
from backtesting import Backtest, Strategy
from backtesting.test import GOOG
import pandas as pd
import numpy as np

# --- Candlestick Pattern Recognition Functions ---

def is_bullish_engulfing(df, i):
    """Checks for a bullish engulfing pattern at index i."""
    if i < 1:
        return False
    prev_candle = df.iloc[i-1]
    curr_candle = df.iloc[i]
    # Previous candle must be bearish, current must be bullish
    if not (prev_candle['Close'] < prev_candle['Open'] and curr_candle['Close'] > curr_candle['Open']):
        return False
    return (curr_candle['Open'] < prev_candle['Close'] and
            curr_candle['Close'] > prev_candle['Open'] and
            (curr_candle['Close'] - curr_candle['Open']) > (prev_candle['Open'] - prev_candle['Close']))

def is_bearish_engulfing(df, i):
    """Checks for a bearish engulfing pattern at index i."""
    if i < 1:
        return False
    prev_candle = df.iloc[i-1]
    curr_candle = df.iloc[i]
    # Previous candle must be bullish, current must be bearish
    if not (prev_candle['Close'] > prev_candle['Open'] and curr_candle['Close'] < curr_candle['Open']):
        return False
    return (curr_candle['Open'] > prev_candle['Close'] and
            curr_candle['Close'] < prev_candle['Open'] and
            (curr_candle['Open'] - curr_candle['Close']) > (prev_candle['Close'] - prev_candle['Open']))

def is_hammer(df, i, body_ratio=0.3, upper_wick_ratio=0.1):
    """Checks for a Hammer pattern (bullish reversal)."""
    candle = df.iloc[i]
    body = abs(candle['Close'] - candle['Open'])
    high = candle['High']
    low = candle['Low']
    total_range = high - low
    if total_range == 0: return False

    lower_wick = min(candle['Open'], candle['Close']) - low
    upper_wick = high - max(candle['Open'], candle['Close'])

    return (candle['Close'] > candle['Open'] and # Must be a green candle
            body / total_range < body_ratio and
            upper_wick / total_range < upper_wick_ratio and
            lower_wick / total_range > (1 - body_ratio - upper_wick_ratio))

def is_shooting_star(df, i, body_ratio=0.3, lower_wick_ratio=0.1):
    """Checks for a Shooting Star pattern (bearish reversal)."""
    candle = df.iloc[i]
    body = abs(candle['Close'] - candle['Open'])
    high = candle['High']
    low = candle['Low']
    total_range = high - low
    if total_range == 0: return False

    upper_wick = high - max(candle['Open'], candle['Close'])
    lower_wick = min(candle['Open'], candle['Close']) - low

    return (candle['Close'] < candle['Open'] and # Must be a red candle
            body / total_range < body_ratio and
            lower_wick / total_range < lower_wick_ratio and
            upper_wick / total_range > (1 - body_ratio - lower_wick_ratio))


class BtcMonthlyPivotMoonCycleStrategy(Strategy):
    risk_perc = 1.0
    min_rr = 5.0
    body_ratio_param = 0.3
    wick_ratio_param = 0.1

    def init(self):
        # Hardcoded Moon Phase Data (YYYY-MM-DD)
        # Source: https://www.timeanddate.com/moon/phases/
        full_moons_2004_2013 = [
            '2004-01-07', '2004-02-06', '2004-03-07', '2004-04-05', '2004-05-05', '2004-06-03', '2004-07-02', '2004-07-31', '2004-08-30', '2004-09-28', '2004-10-28', '2004-11-26', '2004-12-26',
            '2005-01-25', '2005-02-24', '2005-03-25', '2005-04-24', '2005-05-23', '2005-06-22', '2005-07-21', '2005-08-19', '2005-09-18', '2005-10-17', '2005-11-16', '2005-12-15',
            '2006-01-14', '2006-02-13', '2006-03-14', '2006-04-13', '2006-05-13', '2006-06-11', '2006-07-11', '2006-08-09', '2006-09-07', '2006-10-07', '2006-11-05', '2006-12-05',
            '2007-01-03', '2007-02-02', '2007-03-03', '2007-04-02', '2007-05-02', '2007-06-01', '2007-06-30', '2007-07-30', '2007-08-28', '2007-09-26', '2007-10-26', '2007-11-24', '2007-12-24',
            '2008-01-22', '2008-02-21', '2008-03-21', '2008-04-20', '2008-05-20', '2008-06-18', '2008-07-18', '2008-08-16', '2008-09-15', '2008-10-14', '2008-11-13', '2008-12-12',
            '2009-01-11', '2009-02-09', '2009-03-11', '2009-04-09', '2009-05-09', '2009-06-07', '2009-07-07', '2009-08-06', '2009-09-04', '2009-10-04', '2009-11-02', '2009-12-02', '2009-12-31',
            '2010-01-30', '2010-02-28', '2010-03-30', '2010-04-28', '2010-05-27', '2010-06-26', '2010-07-26', '2010-08-24', '2010-09-23', '2010-10-23', '2010-11-21', '2010-12-21',
            '2011-01-19', '2011-02-18', '2011-03-19', '2011-04-18', '2011-05-17', '2011-06-15', '2011-07-15', '2011-08-13', '2011-09-12', '2011-10-12', '2011-11-10', '2011-12-10',
            '2012-01-09', '2012-02-07', '2012-03-08', '2012-04-06', '2012-05-06', '2012-06-04', '2012-07-03', '2012-08-02', '2012-08-31', '2012-09-30', '2012-10-29', '2012-11-28', '2012-12-28',
            '2013-01-27', '2013-02-25', '2013-03-27', '2013-04-25', '2013-05-25', '2013-06-23', '2013-07-22', '2013-08-21',
        ]
        new_moons_2004_2013 = [
            '2004-01-21', '2004-02-20', '2004-03-20', '2004-04-19', '2004-05-19', '2004-06-17', '2004-07-17', '2004-08-16', '2004-09-14', '2004-10-14', '2004-11-12', '2004-12-12',
            '2005-01-10', '2005-02-08', '2005-03-10', '2005-04-08', '2005-05-08', '2005-06-06', '2005-07-06', '2005-08-05', '2005-09-03', '2005-10-03', '2005-11-02', '2005-12-01', '2005-12-31',
            '2006-01-29', '2006-02-28', '2006-03-29', '2006-04-27', '2006-05-27', '2006-06-25', '2006-07-25', '2006-08-23', '2006-09-22', '2006-10-22', '2006-11-20', '2006-12-20',
            '2007-01-19', '2007-02-17', '2007-03-19', '2007-04-17', '2007-05-16', '2007-06-14', '2007-07-14', '2007-08-12', '2007-09-11', '2007-10-11', '2007-11-09', '2007-12-09',
            '2008-01-08', '2008-02-07', '2008-03-07', '2008-04-06', '2008-05-05', '2008-06-03', '2008-07-03', '2008-08-01', '2008-08-30', '2008-09-29', '2008-10-28', '2008-11-27', '2008-12-27',
            '2009-01-26', '2009-02-25', '2009-03-26', '2009-04-25', '2009-05-24', '2009-06-22', '2009-07-22', '2009-08-20', '2009-09-18', '2009-10-18', '2009-11-16', '2009-12-16',
            '2010-01-15', '2010-02-14', '2010-03-15', '2010-04-14', '2010-05-14', '2010-06-12', '2010-07-11', '2010-08-10', '2010-09-08', '2010-10-07', '2010-11-06', '2010-12-05',
            '2011-01-04', '2011-02-03', '2011-03-04', '2011-04-03', '2011-05-03', '2011-06-01', '2011-07-01', '2011-07-30', '2011-08-29', '2011-09-27', '2011-10-26', '2011-11-25', '2011-12-24',
            '2012-01-23', '2012-02-21', '2012-03-22', '2012-04-21', '2012-05-20', '2012-06-19', '2012-07-19', '2012-08-17', '2012-09-16', '2012-10-15', '2012-11-13', '2012-12-13',
            '2013-01-11', '2013-02-10', '2013-03-11', '2013-04-10', '2013-05-10', '2013-06-08', '2013-07-08', '2013-08-06',
        ]

        # Using .data.df is recommended to modify the underlying DataFrame
        self.data.df['is_full_moon'] = self.data.df.index.strftime('%Y-%m-%d').isin(full_moons_2004_2013)
        self.data.df['is_new_moon'] = self.data.df.index.strftime('%Y-%m-%d').isin(new_moons_2004_2013)

        self.full_moon_signal = self.I(lambda x: x, self.data.df['is_full_moon'], plot=False, name="FullMoon")
        self.new_moon_signal = self.I(lambda x: x, self.data.df['is_new_moon'], plot=False, name="NewMoon")


    def next(self):
        current_date = self.data.index[-1]

        # Time-based exit: Close positions on the last day of the month
        if current_date.is_month_end and self.position:
            self.position.close()
            return

        # --- Entry Logic ---
        # Only trade between the 1st and 12th day of the month
        if not 1 <= current_date.day <= 12:
            return

        # Check if a position is already open
        if self.position:
            return

        # Find recent moon events within the valid window
        full_moon_in_window = False
        new_moon_in_window = False

        # Check the last 15 days for a moon event, but only if it happened this month
        for i in range(1, 16):
            if len(self.data.Close) <= i: break

            date_to_check = self.data.index[-i]
            if date_to_check.month == current_date.month and 1 <= date_to_check.day <= 15:
                if self.full_moon_signal[-i]:
                    full_moon_in_window = True
                if self.new_moon_signal[-i]:
                    new_moon_in_window = True

        df = self.data.df
        i = len(df) - 1

        # LONG ENTRY
        if full_moon_in_window:
            is_bullish_pattern = (is_bullish_engulfing(df, i) or
                                  is_hammer(df, i, self.body_ratio_param, self.wick_ratio_param))
            if is_bullish_pattern:
                entry_price = self.data.Close[-1] # Use current close as proxy for next open
                sl = df.iloc[i]['Low'] * 0.999 # SL just beyond the low of the signal candle
                risk = entry_price - sl
                if risk <= 0: return
                tp = entry_price + risk * self.min_rr
                size = (self.equity * (self.risk_perc / 100)) / risk

                if size > 0:
                    self.buy(size=size, sl=sl, tp=tp)

        # SHORT ENTRY
        elif new_moon_in_window:
            is_bearish_pattern = (is_bearish_engulfing(df, i) or
                                  is_shooting_star(df, i, self.body_ratio_param, self.wick_ratio_param))
            if is_bearish_pattern:
                entry_price = self.data.Close[-1] # Use current close as proxy for next open
                sl = df.iloc[i]['High'] * 1.001 # SL just beyond the high of the signal candle
                risk = sl - entry_price
                if risk <= 0: return
                tp = entry_price - risk * self.min_rr
                size = (self.equity * (self.risk_perc / 100)) / risk

                if size > 0:
                    self.sell(size=size, sl=sl, tp=tp)

if __name__ == '__main__':
    # Load or generate data
    data = GOOG.copy()
    # The GOOG dataset is daily, which fits the 1D analysis part of the strategy.
    # Entry confirmation on 4H would require 4H data, but for this example, we'll use 1D.

    # Run backtest
    bt = Backtest(data, BtcMonthlyPivotMoonCycleStrategy, cash=1_000_000, commission=.002)

    # Optimize
    # We can optimize the candlestick pattern parameters
    stats = bt.optimize(
        risk_perc=[0.5, 1.0, 1.5],
        min_rr=[3.0, 5.0, 7.0],
        body_ratio_param=np.arange(0.1, 0.5, 0.1),
        wick_ratio_param=np.arange(0.05, 0.2, 0.05),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.risk_perc > 0 and p.min_rr > 0
    )

    print("Best stats:", stats)

    # Save results
    os.makedirs('results', exist_ok=True)

    # Handle cases where no trades are made
    if stats['# Trades'] > 0:
        result_data = {
            'strategy_name': 'btc_monthly_pivot_moon_cycle',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }
    else:
        result_data = {
            'strategy_name': 'btc_monthly_pivot_moon_cycle',
            'return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0
        }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_data, f, indent=2)

    print("Results saved to results/temp_result.json")

    # Generate plot
    try:
        bt.plot(filename='results/btc_monthly_pivot_moon_cycle.html')
        print("Plot saved to results/btc_monthly_pivot_moon_cycle.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
