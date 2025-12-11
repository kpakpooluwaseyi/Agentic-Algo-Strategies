
import json
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

# --- Utility Functions ---

def pass_through(series):
    """A utility function to pass a pandas Series through self.I()."""
    return series

def generate_synthetic_data(periods=96 * 60, base_price=1.0):
    """
    Generates synthetic 15-minute data with specific patterns for this strategy.
    """
    index = pd.to_datetime(pd.date_range('2023-01-02 00:00', periods=periods, freq='15min'))
    data = pd.DataFrame(index=index)
    data['Open'] = np.nan
    data['High'] = np.nan
    data['Low'] = np.nan
    data['Close'] = np.nan

    price = base_price
    for i in range(len(data)):
        dt = data.index[i]

        # Asia Session (00:00 - 07:45): Consolidation
        if 0 <= dt.hour < 8:
            price_change = np.random.uniform(-0.0005, 0.0005)
            open_price = price
            close_price = open_price + price_change
            high = max(open_price, close_price) + np.random.uniform(0, 0.0002)
            low = min(open_price, close_price) - np.random.uniform(0, 0.0002)

        # UK Session (08:00 - 15:45): Potential Breakout and Reversal
        elif 8 <= dt.hour < 16:
            # Create a liquidity grab scenario on some days
            if dt.hour == 8 and dt.minute == 15 and dt.weekday() < 4: # Tue-Fri
                 # Bearish setup
                if np.random.rand() > 0.5:
                    open_price = price
                    high = price + np.random.uniform(0.001, 0.002) # Grab liquidity above Asia High
                    low = open_price - np.random.uniform(0, 0.0005)
                    close_price = high - np.random.uniform(0.0001, 0.0003)
                # Bullish setup
                else:
                    open_price = price
                    low = price - np.random.uniform(0.001, 0.002) # Grab liquidity below Asia Low
                    high = open_price + np.random.uniform(0, 0.0005)
                    close_price = low + np.random.uniform(0.0001, 0.0003)
            else: # Standard UK session movement
                price_change = np.random.uniform(-0.001, 0.001)
                open_price = price
                close_price = open_price + price_change
                high = max(open_price, close_price) + np.random.uniform(0, 0.0005)
                low = min(open_price, close_price) - np.random.uniform(0, 0.0005)

        # US Session (16:00 onwards): Drift
        else:
            price_change = np.random.uniform(-0.0008, 0.0008)
            open_price = price
            close_price = open_price + price_change
            high = max(open_price, close_price) + np.random.uniform(0, 0.0003)
            low = min(open_price, close_price) - np.random.uniform(0, 0.0003)

        data.iloc[i] = [open_price, high, low, close_price]
        price = close_price

    data['Volume'] = np.random.randint(100, 1000, periods)
    data.ffill(inplace=True)
    return data

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df.index.hour
    df['is_asia'] = (df['hour'] >= 0) & (df['hour'] < 8)
    df['is_uk'] = (df['hour'] >= 8) & (df['hour'] < 16)
    asia_session_data = df[df['is_asia']].groupby(df.index.date).agg(
        asia_high=('High', 'max'),
        asia_low=('Low', 'min')
    )
    df['asia_high'] = pd.Series(df.index.date, index=df.index).map(asia_session_data['asia_high'])
    df['asia_low'] = pd.Series(df.index.date, index=df.index).map(asia_session_data['asia_low'])
    df['asia_high'] = df['asia_high'].ffill()
    df['asia_low'] = df['asia_low'].ffill()
    df['asia_range_pct'] = ((df['asia_high'] - df['asia_low']) / df['asia_low']) * 100
    df.drop(columns=['hour'], inplace=True)
    df.dropna(inplace=True)
    return df

# --- Strategy Class ---

class AsiaLiquidityReversalUkSessionStrategy(Strategy):
    asia_range_max_pct = 2.0
    sl_buffer_pct = 0.1

    def init(self):
        self.is_uk = self.I(pass_through, self.data.df['is_uk'].values, name='is_uk')
        self.asia_high = self.I(pass_through, self.data.df['asia_high'].values, name='asia_high')
        self.asia_low = self.I(pass_through, self.data.df['asia_low'].values, name='asia_low')
        self.asia_range_pct = self.I(pass_through, self.data.df['asia_range_pct'].values, name='asia_range_pct')

    def is_bearish_engulfing(self):
        if len(self.data.Close) < 2: return False
        prev_open, prev_close = self.data.Open[-2], self.data.Close[-2]
        curr_open, curr_close = self.data.Open[-1], self.data.Close[-1]
        return prev_close > prev_open and curr_open > prev_close and curr_close < prev_open

    def is_bullish_engulfing(self):
        if len(self.data.Close) < 2: return False
        prev_open, prev_close = self.data.Open[-2], self.data.Close[-2]
        curr_open, curr_close = self.data.Open[-1], self.data.Close[-1]
        return prev_close < prev_open and curr_open < prev_close and curr_close > prev_open

    def next(self):
        if len(self.data.Close) < 2 or self.position:
            return

        is_valid_session = self.is_uk[-1] == 1
        is_valid_range = self.asia_range_pct[-1] < self.asia_range_max_pct
        if not is_valid_session or not is_valid_range:
            return

        # Short Entry
        liquidity_grab_short = self.data.High[-2] > self.asia_high[-1]
        if liquidity_grab_short and self.is_bearish_engulfing():
            sl_price = self.data.High[-2] * (1 + self.sl_buffer_pct / 100)
            tp_price = self.asia_low[-1]
            if sl_price > self.data.Close[-1] and tp_price < self.data.Close[-1]:
                self.sell(sl=sl_price, tp=tp_price)
            return

        # Long Entry
        liquidity_grab_long = self.data.Low[-2] < self.asia_low[-1]
        if liquidity_grab_long and self.is_bullish_engulfing():
            sl_price = self.data.Low[-2] * (1 - self.sl_buffer_pct / 100)
            tp_price = self.asia_high[-1]
            if sl_price < self.data.Close[-1] and tp_price > self.data.Close[-1]:
                self.buy(sl=sl_price, tp=tp_price)

# --- Main Execution Block ---

if __name__ == '__main__':
    data = generate_synthetic_data()
    data = preprocess_data(data.copy())

    bt = Backtest(data, AsiaLiquidityReversalUkSessionStrategy, cash=100_000, commission=.002)

    stats = bt.optimize(
        asia_range_max_pct=np.arange(0.5, 3.1, 0.5).tolist(),
        sl_buffer_pct=np.arange(0.05, 0.51, 0.05).tolist(),
        maximize='Sharpe Ratio'
    )

    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON output
    sanitized_stats = {
        'strategy_name': 'asia_liquidity_reversal_uk_session',
        'return': float(stats.get('Return [%]', 0)),
        'sharpe': float(stats.get('Sharpe Ratio', 0)),
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)),
        'win_rate': float(stats.get('Win Rate [%]', 0)),
        'total_trades': int(stats.get('# Trades', 0))
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=2)

    print("Optimization complete. Best run stats saved to results/temp_result.json")
    print(stats)

    try:
        bt.plot(filename='results/plot.html', open_browser=False)
        print("Plot saved to results/plot.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
