import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

# =====================================================================================
# SYNTHETIC DATA GENERATION
# =====================================================================================
def generate_forex_data(days=365):
    """
    Generates synthetic 24-hour OHLCV data for a 15-minute timeframe.
    """
    rng = pd.date_range('2022-01-01', periods=days * 24 * 4, freq='15min', tz='UTC')
    price_change = np.random.randn(len(rng)) * 0.001
    price = 1.1000 + np.cumsum(price_change)
    price += np.sin(np.arange(len(rng)) / (24 * 4 * 7) * np.pi * 2) * 0.01
    is_session_open = (rng.hour == 8) | (rng.hour == 13)
    price[is_session_open] *= (1 + np.random.randn(np.sum(is_session_open)) * 0.005)

    df = pd.DataFrame(index=rng)
    df['Open'] = price
    df['High'] = df['Open'] + np.random.rand(len(rng)) * 0.001
    df['Low'] = df['Open'] - np.random.rand(len(rng)) * 0.001
    df['Close'] = df['Open'] + (np.random.rand(len(rng)) - 0.5) * 0.002
    df['Volume'] = np.random.randint(100, 1000, size=len(rng))

    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

    return df

# =====================================================================================
# DATA PREPROCESSING FOR SESSION ANALYSIS
# =====================================================================================
def preprocess_data(df):
    """
    Adds session information and calculates the daily Asia session high and low.
    """
    df['session'] = 'Other'
    df.loc[df.index.hour.isin(range(0, 8)), 'session'] = 'Asia'
    df.loc[df.index.hour.isin(range(8, 16)), 'session'] = 'London'
    df['is_ny_session'] = df.index.hour.isin(range(13, 21))

    asia_session = df[df['session'] == 'Asia']
    daily_asia_high = asia_session['High'].resample('D').max()
    daily_asia_low = asia_session['Low'].resample('D').min()

    df['Asia_High'] = df.index.normalize().map(daily_asia_high)
    df['Asia_Low'] = df.index.normalize().map(daily_asia_low)

    df['Asia_High'] = df['Asia_High'].ffill()
    df['Asia_Low'] = df['Asia_Low'].ffill()

    df['Asia_Range'] = df['Asia_High'] - df['Asia_Low']
    df.dropna(subset=['Asia_High', 'Asia_Low'], inplace=True)

    return df

# =====================================================================================
# STRATEGY IMPLEMENTATION
# =====================================================================================
class AsiaLiquidityGrabReversalStrategy(Strategy):
    """
    Implements the Asia Liquidity Grab Reversal strategy with corrected logic.
    """
    asia_range_pct_max = 0.02
    risk_reward_ratio = 2.0

    def init(self):
        # No stateful flags needed; logic is now event-driven.
        pass

    def next(self):
        # Ensure we have enough data for lookbacks
        if len(self.data) < 2:
            return

        current_session = self.data.df['session'].iloc[-1]
        is_london = current_session == 'London'
        is_ny = self.data.df['is_ny_session'].iloc[-1]

        if not (is_london or is_ny):
            return

        asia_high = self.data.df['Asia_High'].iloc[-1]
        asia_low = self.data.df['Asia_Low'].iloc[-1]
        asia_range = asia_high - asia_low

        if asia_low > 0 and (asia_range / asia_low) > self.asia_range_pct_max:
            return

        if self.position:
            return

        # --- REFACTORED SHORT ENTRY LOGIC ---
        # 1. Identify a Bearish Engulfing Reversal Pattern
        #    - Current candle [-1] is bearish
        #    - Previous candle [-2] was bullish (a reversal)
        #    - Current candle's body engulfs previous candle's body
        is_bearish_reversal = (self.data.Close[-1] < self.data.Open[-1] and
                               self.data.Close[-2] > self.data.Open[-2] and
                               self.data.Open[-1] > self.data.Close[-2] and
                               self.data.Close[-1] < self.data.Open[-2])

        if is_bearish_reversal:
            # 2. Confirm the PREVIOUS candle [-2] grabbed liquidity
            grab_candle_high = self.data.High[-2]
            if grab_candle_high > asia_high:
                # 3. Place trade with CORRECT stop-loss
                sl = grab_candle_high  # SL is beyond the grab high
                entry_price = self.data.Close[-1]
                tp1 = asia_low

                if tp1 >= entry_price: return # Validate TP

                risk = sl - entry_price
                if risk <= 0: return
                tp2 = entry_price - (risk * self.risk_reward_ratio)

                self.sell(size=0.5, sl=sl, tp=tp1)
                self.sell(size=0.5, sl=sl, tp=tp2)

        # --- REFACTORED LONG ENTRY LOGIC ---
        # 1. Identify a Bullish Engulfing Reversal Pattern
        #    - Current candle [-1] is bullish
        #    - Previous candle [-2] was bearish (a reversal)
        #    - Current candle's body engulfs previous candle's body
        is_bullish_reversal = (self.data.Close[-1] > self.data.Open[-1] and
                               self.data.Close[-2] < self.data.Open[-2] and
                               self.data.Open[-1] < self.data.Close[-2] and
                               self.data.Close[-1] > self.data.Open[-2])

        if is_bullish_reversal:
            # 2. Confirm the PREVIOUS candle [-2] grabbed liquidity
            grab_candle_low = self.data.Low[-2]
            if grab_candle_low < asia_low:
                # 3. Place trade with CORRECT stop-loss
                sl = grab_candle_low # SL is beyond the grab low
                entry_price = self.data.Close[-1]
                tp1 = asia_high

                if tp1 <= entry_price: return # Validate TP

                risk = entry_price - sl
                if risk <= 0: return
                tp2 = entry_price + (risk * self.risk_reward_ratio)

                self.buy(size=0.5, sl=sl, tp=tp1)
                self.buy(size=0.5, sl=sl, tp=tp2)

# =====================================================================================
# BACKTEST EXECUTION
# =====================================================================================
if __name__ == '__main__':
    print("Generating synthetic 24-hour data...")
    data = generate_forex_data(days=180)
    print("Preprocessing data for session analysis...")
    data = preprocess_data(data)

    print("Initializing backtest...")
    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=100000, commission=.0002)

    print("Optimizing strategy...")
    stats = bt.optimize(
        asia_range_pct_max=[0.01, 0.02, 0.03],
        risk_reward_ratio=[1.5, 2.0, 2.5],
        maximize='Sharpe Ratio'
    )

    print("Best run stats:")
    print(stats)

    import os
    os.makedirs('results', exist_ok=True)

    if stats['# Trades'] > 0:
        result_data = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }
    else:
        result_data = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0
        }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    print("Results saved to results/temp_result.json")

    print("Generating plot...")
    bt.plot(filename="results/asia_liquidity_grab_reversal.html", open_browser=False)
    print("Plot saved to results/asia_liquidity_grab_reversal.html")
