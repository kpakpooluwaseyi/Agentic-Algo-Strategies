from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os

def generate_forex_data(days=200):
    """Generates synthetic 24-hour forex data with session-specific volatility."""
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range(start='2023-01-01', periods=days * 24 * 4, freq='15min')

    price = 1.1000
    prices = []
    for date in dates:
        if date.hour == 8 and date.minute < 30: # London open spike
            volatility = rng.uniform(0.0006, 0.0012)
            drift = 0
        elif 8 <= date.hour < 16: # London session
            volatility = rng.uniform(0.0003, 0.0008)
            drift = 0.00002
        elif 0 <= date.hour < 8: # Asia session
            volatility = rng.uniform(0.0001, 0.0004)
            drift = 0.00001
        else: # NY session
            volatility = rng.uniform(0.0002, 0.0006)
            drift = -0.00001

        price += drift + rng.normal(0, volatility)
        prices.append(price)

    df = pd.DataFrame(index=dates)
    df['Open'] = [p - rng.uniform(0, 0.0002) for p in prices]
    df['Close'] = prices
    df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0, 0.0005, size=len(df))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0, 0.0005, size=len(df))
    df['Volume'] = rng.integers(100, 1000, size=len(df))

    return df

def preprocess_data(df: pd.DataFrame):
    """
    Adds session information and Asia range to the DataFrame.
    Returns the processed DataFrame and the integer code for the London session.
    """
    df = df.copy()
    times = pd.Series(df.index.time, index=df.index).astype(str)

    df['session'] = 'NY'
    df.loc[times.between('00:00:00', '08:00:00', inclusive='left'), 'session'] = 'Asia'
    df.loc[times.between('08:00:00', '16:00:00', inclusive='left'), 'session'] = 'London'

    df['session'] = df['session'].astype('category')
    session_cats = list(df['session'].cat.categories)
    london_code = session_cats.index('London')

    asia_sessions = df[df['session'] == 'Asia']
    if not asia_sessions.empty:
        asia_range = asia_sessions.groupby(asia_sessions.index.date).agg(H_Asia=('High', 'max'), L_Asia=('Low', 'min'))
        date_series = pd.Series(df.index.date, index=df.index)
        df['H_Asia'] = date_series.map(asia_range['H_Asia'])
        df['L_Asia'] = date_series.map(asia_range['L_Asia'])
        df[['H_Asia', 'L_Asia']] = df[['H_Asia', 'L_Asia']].ffill()
    else:
        df['H_Asia'], df['L_Asia'] = np.nan, np.nan

    # Convert session to integer codes AFTER using it for filtering
    df['session'] = df['session'].cat.codes

    df.dropna(inplace=True)
    return df, london_code

def is_bearish_engulfing(data):
    """Detects a bearish engulfing pattern."""
    if len(data.Open) < 2: return False
    prev_open, prev_close = data.Open[-2], data.Close[-2]
    curr_open, curr_close = data.Open[-1], data.Close[-1]
    return curr_close < curr_open and curr_open > prev_close and curr_close < prev_open

class AsiaLiquidityGrabReversalStrategy(Strategy):
    asia_range_threshold = 0.02
    confirmation_window = 3
    london_session_code = -1

    def init(self):
        self.session_codes = self.data.df['session'].to_numpy()
        self.H_Asia_values = self.data.df['H_Asia'].to_numpy()
        self.L_Asia_values = self.data.df['L_Asia'].to_numpy()

        self.breakout_bar_index = -1
        self.breakout_peak = 0

    def next(self):
        i = len(self.data.Close) - 1
        if i < 1: return

        if self.session_codes[i] != self.london_session_code:
            self.breakout_bar_index = -1
            return

        h_asia, l_asia = self.H_Asia_values[i], self.L_Asia_values[i]
        asia_range = h_asia - l_asia
        if asia_range == 0 or asia_range / self.data.Close[-1] > self.asia_range_threshold:
            self.breakout_bar_index = -1
            return

        if self.breakout_bar_index == -1 and self.data.High[-1] > h_asia:
            self.breakout_bar_index = i
            self.breakout_peak = self.data.High[-1]

        if self.breakout_bar_index != -1:
            self.breakout_peak = max(self.breakout_peak, self.data.High[-1])

            if i - self.breakout_bar_index > self.confirmation_window:
                self.breakout_bar_index = -1
                return

            if is_bearish_engulfing(self.data):
                if not self.position:
                    sl = self.breakout_peak
                    tp = l_asia
                    if sl > self.data.Close[-1] and tp < self.data.Close[-1]:
                        self.sell(sl=sl, tp=tp)

                self.breakout_bar_index = -1

if __name__ == '__main__':
    data, london_code = preprocess_data(generate_forex_data(days=200))

    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=100000, commission=.0002)

    stats = bt.optimize(
        asia_range_threshold=[i/1000 for i in range(5, 40, 5)],
        confirmation_window=range(2, 6),
        london_session_code=[london_code],
        maximize='Sharpe Ratio'
    )

    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        stats_dict = dict(stats)
        stats_dict.pop('_strategy', None)
        stats_dict.pop('_equity_curve', None)
        stats_dict.pop('_trades', None)

        json.dump({
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': stats_dict.get('Return [%]', 0.0),
            'sharpe': stats_dict.get('Sharpe Ratio', 0.0),
            'max_drawdown': stats_dict.get('Max. Drawdown [%]', 0.0),
            'win_rate': stats_dict.get('Win Rate [%]', 0.0),
            'total_trades': int(stats_dict.get('# Trades', 0))
        }, f, indent=2)

    bt.plot(filename='results/asia_liquidity_grab_reversal.html', open_browser=False)
    print("Backtest complete. Results saved to results/temp_result.json and plot saved to results/asia_liquidity_grab_reversal.html")
