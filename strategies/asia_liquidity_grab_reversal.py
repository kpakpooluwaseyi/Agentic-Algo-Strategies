
import pandas as pd
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from backtesting.test import EURUSD

# Helper function to pass pre-processed columns through to the strategy
def pass_through(series):
    return series

# Data pre-processing function
def preprocess_data(df):
    """Adds session information and Asia session high/low to the DataFrame."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        return df # Return original df if index is not datetime

    df['hour'] = df.index.hour

    conditions = [
        (df['hour'] >= 0) & (df['hour'] < 8),
        (df['hour'] >= 8) & (df['hour'] < 16)
    ]
    choices = ['Asia', 'UK']
    df['session_str'] = np.select(conditions, choices, default='Other')
    df['session'] = pd.Categorical(df['session_str'], categories=['Other', 'Asia', 'UK'], ordered=True).codes

    asia_session_data = df[df['session'] == 1]
    daily_stats = asia_session_data.groupby(asia_session_data.index.date).agg(
        HOA=('High', 'max'),
        LOA=('Low', 'min')
    )

    df['date'] = df.index.date
    df = pd.merge(df, daily_stats, left_on='date', right_index=True, how='left')
    df = df.drop(columns=['date', 'session_str'])

    df['Asia_Range'] = df['HOA'] - df['LOA']

    df = df.ffill()
    df = df.dropna()

    return df

class AsiaLiquidityGrabReversalStrategy(Strategy):
    asia_range_threshold = 0.002 # Adjusted for realistic data
    risk_pct = 0.02
    sl_buffer = 0.0005 # Adjusted
    rr_multiplier = 2.0

    def init(self):
        self.tp1_price_short = None
        self.tp1_price_long = None

        self.session = self.I(pass_through, self.data.df['session'].values, name="Session")
        self.hoa = self.I(pass_through, self.data.df['HOA'].values, name="HOA")
        self.loa = self.I(pass_through, self.data.df['LOA'].values, name="LOA")
        self.asia_range = self.I(pass_through, self.data.df['Asia_Range'].values, name="Asia_Range")

    def next(self):
        if len(self.data.Close) < 2:
            return

        if self.position.is_short and self.tp1_price_short and self.data.Low[-1] <= self.tp1_price_short:
            self.position.close(portion=0.5)
            self.tp1_price_short = None

        if self.position.is_long and self.tp1_price_long and self.data.High[-1] >= self.tp1_price_long:
            self.position.close(portion=0.5)
            self.tp1_price_long = None

        if self.position:
            return

        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_open = self.data.Open[-1]
        current_close = self.data.Close[-1]
        prev_open = self.data.Open[-2]
        prev_close = self.data.Close[-2]
        session_code = self.session[-1]
        hoa = self.hoa[-1]
        loa = self.loa[-1]
        asia_range = self.asia_range[-1]
        entry_price = self.data.Close[-1]
        is_uk_session = session_code == 2

        # --- SHORT ENTRY LOGIC ---
        is_range_ok = asia_range < (current_close * self.asia_range_threshold)
        liquidity_grab = current_high > hoa
        is_bearish_engulfing = current_close < current_open and current_open > prev_close and current_close <= prev_open
        close_below_hoa = current_close < hoa

        if is_uk_session and is_range_ok and liquidity_grab and is_bearish_engulfing and close_below_hoa:
            sl = current_high + self.sl_buffer
            tp1 = loa
            risk = abs(entry_price - sl)
            if risk > 0:
                tp2 = entry_price - risk * self.rr_multiplier
                if sl > entry_price and tp2 < entry_price:
                    size = (self.equity * self.risk_pct) / risk
                    self.sell(sl=sl, tp=tp2, size=int(size))
                    self.tp1_price_short = tp1

        # --- LONG ENTRY LOGIC (REVERSE) ---
        is_range_ok = asia_range < (current_close * self.asia_range_threshold)
        liquidity_grab = current_low < loa
        is_bullish_engulfing = current_close > current_open and current_open < prev_close and current_close >= prev_open
        close_above_loa = current_close > loa

        if is_uk_session and is_range_ok and liquidity_grab and is_bullish_engulfing and close_above_loa:
            sl = current_low - self.sl_buffer
            tp1 = hoa
            risk = abs(entry_price - sl)
            if risk > 0:
                tp2 = entry_price + risk * self.rr_multiplier
                if sl < entry_price and tp2 > entry_price:
                    size = (self.equity * self.risk_pct) / risk
                    self.buy(sl=sl, tp=tp2, size=int(size))
                    self.tp1_price_long = tp1

if __name__ == '__main__':
    # Use realistic EURUSD data and resample to 15min
    data = EURUSD.copy()
    data = data['2018':'2020']
    data = data.resample('15min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

    data = preprocess_data(data)

    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=100_000, commission=.0002)

    stats = bt.optimize(
        asia_range_threshold=np.arange(0.001, 0.005, 0.001).tolist(),
        risk_pct=np.arange(0.01, 0.04, 0.01).tolist(),
        sl_buffer=np.arange(0.0001, 0.0006, 0.0001).tolist(),
        rr_multiplier=np.arange(1.5, 3.0, 0.5).tolist(),
        maximize='Sharpe Ratio'
    )

    os.makedirs('results', exist_ok=True)

    def sanitize(value):
        if pd.isna(value): return None
        if isinstance(value, (np.int64, np.int32)): return int(value)
        if isinstance(value, (np.float64, np.float32)): return float(value)
        return value

    try:
        stats_dict = dict(stats)
        for key in ['_strategy', '_equity_curve', '_trades']: stats_dict.pop(key, None)

        with open('results/temp_result.json', 'w') as f:
            json.dump({
                'strategy_name': 'asia_liquidity_grab_reversal',
                'return': sanitize(stats_dict.get('Return [%]')),
                'sharpe': sanitize(stats_dict.get('Sharpe Ratio')),
                'max_drawdown': sanitize(stats_dict.get('Max. Drawdown [%]')),
                'win_rate': sanitize(stats_dict.get('Win Rate [%]')),
                'total_trades': sanitize(stats_dict.get('# Trades'))
            }, f, indent=2)
        print("Successfully saved results to results/temp_result.json")
    except Exception as e:
        print(f"Error saving results: {e}")

    try:
        plot_filename = 'results/asia_liquidity_grab_reversal.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Successfully generated plot: {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
