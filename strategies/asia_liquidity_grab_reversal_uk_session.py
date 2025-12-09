from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os

# --- Synthetic Data Generation ---
def generate_synthetic_data():
    """
    Generates synthetic 15-minute data to trigger both SHORT and LONG scenarios.
    """
    dates = pd.date_range(start='2004-08-11', periods=4 * 24 * 5, freq='15min', tz='UTC')
    df = pd.DataFrame(index=dates)
    np.random.seed(42)
    price_changes = np.random.randn(len(df)) * 0.1
    df['Close'] = 100 + np.cumsum(price_changes)
    df['Open'] = df['Close'].shift(1)

    # --- SHORT SCENARIO (Day 2) ---
    day2_start, day2_end = 96, 192
    day2_indices = df.index[day2_start:day2_end]
    asia_mask2 = (day2_indices.hour >= 0) & (day2_indices.hour < 8)
    asia_low2, asia_high2 = 100, 100.5
    df.loc[day2_indices[asia_mask2], 'Close'] = np.linspace(asia_high2, asia_low2, np.sum(asia_mask2))

    uk_mask2 = (day2_indices.hour >= 8)
    uk_indices2 = day2_indices[uk_mask2]
    grab_candle_idx = uk_indices2[4]
    confirm_candle_idx = uk_indices2[5]

    df.loc[grab_candle_idx, 'High'] = asia_high2 + 0.1 # The grab
    df.loc[confirm_candle_idx, 'Open'] = asia_high2 + 0.05
    df.loc[confirm_candle_idx, 'Close'] = asia_high2 - 0.1 # Close back below
    df.loc[confirm_candle_idx, 'High'] = asia_high2 + 0.1
    df.loc[confirm_candle_idx, 'Low'] = asia_high2 - 0.15
    df.loc[grab_candle_idx, 'Open'] = df.loc[confirm_candle_idx, 'Close'] + 0.02
    df.loc[grab_candle_idx, 'Close'] = df.loc[confirm_candle_idx, 'Open'] - 0.02


    # --- LONG SCENARIO (Day 4) ---
    day4_start, day4_end = 288, 384
    day4_indices = df.index[day4_start:day4_end]
    asia_mask4 = (day4_indices.hour >= 0) & (day4_indices.hour < 8)
    asia_low4, asia_high4 = 98.0, 98.5
    df.loc[day4_indices[asia_mask4], 'Close'] = np.linspace(asia_low4, asia_high4, np.sum(asia_mask4))

    uk_mask4 = (day4_indices.hour >= 8)
    uk_indices4 = day4_indices[uk_mask4]
    grab_candle_idx_long = uk_indices4[4]
    confirm_candle_idx_long = uk_indices4[5]

    df.loc[grab_candle_idx_long, 'Low'] = asia_low4 - 0.1 # The grab
    df.loc[confirm_candle_idx_long, 'Open'] = asia_low4 - 0.05
    df.loc[confirm_candle_idx_long, 'Close'] = asia_low4 + 0.1 # Close back above
    df.loc[confirm_candle_idx_long, 'High'] = asia_low4 + 0.15
    df.loc[confirm_candle_idx_long, 'Low'] = asia_low4 - 0.1
    df.loc[grab_candle_idx_long, 'Open'] = df.loc[confirm_candle_idx_long, 'Close'] - 0.02
    df.loc[grab_candle_idx_long, 'Close'] = df.loc[confirm_candle_idx_long, 'Open'] + 0.02


    df['Open'] = df['Close'].shift(1)
    df.bfill(inplace=True)
    df['High'] = df[['Open', 'Close']].max(axis=1) + 0.05
    df['Low'] = df[['Open', 'Close']].min(axis=1) - 0.05

    return df

# --- Pre-processing ---
def preprocess_data(df):
    """A robust preprocessing function that preserves the DatetimeIndex."""
    df['hour'] = df.index.hour
    is_asia = (df['hour'] >= 0) & (df['hour'] < 8)
    is_uk = (df['hour'] >= 8) & (df['hour'] < 16)

    df['session_code'] = np.select([is_asia, is_uk], [1, 2], default=0).astype(int)

    df['date'] = df.index.date
    asia_stats = df[df['session_code'] == 1].groupby('date').agg(
        hoa=('High', 'max'),
        loa=('Low', 'min')
    )

    df['hoa'] = df['date'].map(asia_stats['hoa'])
    df['loa'] = df['date'].map(asia_stats['loa'])

    df['hoa'] = df['hoa'].ffill()
    df['loa'] = df['loa'].ffill()

    df['asia_range_pct'] = ((df['hoa'] - df['loa']) / df['loa']) * 100

    df = df.drop(columns=['hour', 'date'])
    return df.dropna()

# --- Strategy Definition ---
class AsiaLiquidityGrabReversalUkSessionStrategy(Strategy):
    asia_range_max_pct = 2.0
    confirmation_window = 2

    def init(self):
        self.bar_of_grab = -1
        self.grab_type = None

    def next(self):
        current_bar = len(self.data) - 1

        if self.data.index[-1].date() != self.data.index[-2].date():
            self.bar_of_grab = -1; self.grab_type = None

        if self.data.session_code[-1] != 2: return

        if not self.data.asia_range_pct[-1] < self.asia_range_max_pct: return

        if self.grab_type is None:
            if self.data.High[-1] > self.data.hoa[-1]:
                self.grab_type = "high"; self.bar_of_grab = current_bar
            elif self.data.Low[-1] < self.data.loa[-1]:
                self.grab_type = "low"; self.bar_of_grab = current_bar

        if self.grab_type is not None:
            if current_bar - self.bar_of_grab > self.confirmation_window:
                self.grab_type = None; return

            if self.grab_type == "high":
                is_reversal = self.data.Close[-1] < self.data.hoa[-1]
                if is_reversal and not self.position:
                    sl = self.data.High[-1] * 1.001
                    tp = self.data.loa[-1]
                    self.sell(sl=sl, tp=tp)
                self.grab_type = None

            elif self.grab_type == "low":
                is_reversal = self.data.Close[-1] > self.data.loa[-1]
                if is_reversal and not self.position:
                    sl = self.data.Low[-1] * 0.999
                    tp = self.data.hoa[-1]
                    self.buy(sl=sl, tp=tp)
                self.grab_type = None

if __name__ == '__main__':
    data = generate_synthetic_data()
    data = preprocess_data(data)
    os.makedirs('results', exist_ok=True)

    bt = Backtest(data, AsiaLiquidityGrabReversalUkSessionStrategy, cash=10000, commission=.002)

    stats = bt.optimize(
        asia_range_max_pct=[1.0, 2.0, 3.0],
        confirmation_window=range(1, 4),
        maximize='Sharpe Ratio',
        max_tries=200
    )

    print(stats)

    with open('results/temp_result.json', 'w') as f:
        output = {
            'strategy_name': 'asia_liquidity_grab_reversal_uk_session',
            'return': stats.get('Return [%]'),
            'sharpe': stats.get('Sharpe Ratio'),
            'max_drawdown': stats.get('Max. Drawdown [%]'),
            'win_rate': stats.get('Win Rate [%]'),
            'total_trades': stats.get('# Trades')
        }
        for key, value in output.items():
            if isinstance(value, float) and np.isnan(value): output[key] = None
        json.dump(output, f, indent=2)

    print("Backtest complete. Results saved to results/temp_result.json")
    try:
        bt.plot(filename='results/asia_liquidity_grab.html', open_browser=False)
        print("Plot saved to results/asia_liquidity_grab.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
