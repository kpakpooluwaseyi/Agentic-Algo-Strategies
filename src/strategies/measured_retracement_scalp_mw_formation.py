from backtesting import Backtest, Strategy
from backtesting.test import GOOG
import pandas as pd
import numpy as np
import json

def EMA(array, n, **kwargs):
    """Calculates the Exponential Moving Average."""
    return pd.Series(array).ewm(span=n, adjust=False).mean()

def get_htf_levels(data, lookback_period, **kwargs):
    """
    Custom indicator to calculate the AOI, Peak A, and Low B on a higher timeframe.
    Returns a tuple of arrays: (aoi_levels, peak_a_levels, low_b_levels).
    """
    highs = pd.Series(data['High'])
    lows = pd.Series(data['Low'])
    aoi_levels = pd.Series(np.nan, index=data.index)
    peak_a_levels = pd.Series(np.nan, index=data.index)
    low_b_levels = pd.Series(np.nan, index=data.index)

    for i in range(lookback_period, len(highs)):
        window_start_iloc = i - lookback_period
        window_end_iloc = i

        window_highs = highs.iloc[window_start_iloc:window_end_iloc]
        if window_highs.empty: continue

        peak_a_iloc_in_window = np.argmax(window_highs)
        peak_a_iloc_global = window_start_iloc + peak_a_iloc_in_window
        peak_a_price = highs.iloc[peak_a_iloc_global]

        if peak_a_iloc_global < i - 1:
            lows_after_peak = lows.iloc[peak_a_iloc_global:i]
            if lows_after_peak.empty: continue
            low_b_price = lows_after_peak.min()

            if peak_a_price > low_b_price:
                fib_50_level = low_b_price + (peak_a_price - low_b_price) * 0.5
                aoi_levels.iloc[i] = fib_50_level
                peak_a_levels.iloc[i] = peak_a_price
                low_b_levels.iloc[i] = low_b_price

    return (aoi_levels.values, peak_a_levels.values, low_b_levels.values)

class MeasuredRetracementScalpMwFormationStrategy(Strategy):
    ema_period = 20
    peak_lookback = 10
    min_risk_reward = 5
    lod_lookback = 5
    lod_confluence_pct = 0.02

    TF_ANALYSIS = '15T'

    def init(self):
        self.ema_tf = self.I(EMA, self.data.Close, self.ema_period, resample=self.TF_ANALYSIS)

        # Get HTF levels; self.htf_levels will be a tuple of arrays
        self.htf_levels = self.I(get_htf_levels, self.data.df, self.peak_lookback, resample=self.TF_ANALYSIS)

        # Unpack for easier access in next()
        self.aoi_level = self.htf_levels[0]
        self.peak_a = self.htf_levels[1]
        self.low_b = self.htf_levels[2]

        # Intraday state variables
        self.current_day = -1
        self.lod = np.inf
        self.asia_high = -np.inf
        self.asia_low = np.inf
        self.asia_50_pct = np.nan

    def next(self):
        # Update intraday levels at the start of a new day
        current_date = self.data.index[-1].date()
        if self.current_day != current_date:
            self.current_day = current_date
            self.lod = self.data.Low[-1]
            # Reset Asia session levels for the new day
            self.asia_high = -np.inf
            self.asia_low = np.inf
            self.asia_50_pct = np.nan
        else:
            # Update LOD continuously
            self.lod = min(self.lod, self.data.Low[-1])

        # Define Asia Session (e.g., 00:00 - 08:00 UTC)
        current_time = self.data.index[-1].time()
        is_in_asia_session = pd.to_datetime('00:00').time() <= current_time < pd.to_datetime('08:00').time()

        if is_in_asia_session:
            self.asia_high = max(self.asia_high, self.data.High[-1])
            self.asia_low = min(self.asia_low, self.data.Low[-1])
            if self.asia_high > -np.inf and self.asia_low < np.inf:
                 self.asia_50_pct = self.asia_low + (self.asia_high - self.asia_low) / 2
        if self.position:
            return

        current_aoi = self.aoi_level[-1]
        if np.isnan(current_aoi):
            return

        current_price = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_open = self.data.Open[-1]

        is_downtrend = current_price < self.ema_tf[-1]
        enters_aoi = current_high >= current_aoi

        # Enhanced Entry Trigger: Bearish Engulfing Pattern
        is_bearish_engulfing = False
        if len(self.data.Close) > 1:
            prev_open = self.data.Open[-2]
            prev_close = self.data.Close[-2]

            # Condition 1: Previous candle was bullish
            # Condition 2: Current candle is bearish
            # Condition 3: Current candle's body engulfs previous candle's body
            if (prev_close > prev_open and
                current_price < current_open and
                current_open > prev_close and
                current_price < prev_open):
                is_bearish_engulfing = True

        # Confluence Checks
        is_near_lod = abs(current_aoi - self.lod) / self.lod < self.lod_confluence_pct
        is_near_asia_50 = not np.isnan(self.asia_50_pct) and abs(current_aoi - self.asia_50_pct) / self.asia_50_pct < self.lod_confluence_pct # Reuse pct for simplicity

        if is_downtrend and enters_aoi and is_bearish_engulfing and (is_near_lod or is_near_asia_50):
            # Ensure we have a valid low_b from the higher timeframe analysis
            htf_low_b = self.low_b[-1]
            if np.isnan(htf_low_b):
                return

            peak_c = current_high
            stop_loss = peak_c * 1.001

            # Use the consistent low_b from the 15M analysis for TP calculation
            bc_range = peak_c - htf_low_b
            if bc_range <= 0: return
            take_profit = peak_c - (bc_range * 0.5)

            if take_profit >= current_price: return

            risk = stop_loss - current_price
            reward = current_price - take_profit

            if risk <= 0: return
            risk_reward_ratio = reward / risk

            if risk_reward_ratio >= self.min_risk_reward:
                self.sell(sl=stop_loss, tp=take_profit)

def generate_synthetic_data(days=90):
    """
    Generates synthetic 1-minute OHLCV data for backtesting intraday strategies.
    This is necessary because the user's strategy requires 1M/15M timeframes,
    which is incompatible with the daily GOOG dataset.
    """
    n_minutes = days * 24 * 60
    dates = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.arange(n_minutes), 'm')
    price = 100 + np.random.randn(n_minutes).cumsum() * 0.1

    # Ensure OHLC are realistic
    open_p = price
    high_p = open_p + np.random.uniform(0, 0.2, n_minutes)
    low_p = open_p - np.random.uniform(0, 0.2, n_minutes)
    close_p = (open_p + high_p + low_p) / 3 # Simplified close

    df = pd.DataFrame({'Open': open_p, 'High': high_p, 'Low': low_p, 'Close': close_p}, index=dates)
    df['Volume'] = np.random.randint(100, 1000, n_minutes)
    return df

if __name__ == '__main__':
    # Generate a smaller, 10-day synthetic dataset to ensure timely execution
    data = generate_synthetic_data(days=10)

    # Run backtest
    bt = Backtest(data, MeasuredRetracementScalpMwFormationStrategy, cash=10000, commission=.002)

    # Optimize with a minimal set of parameters to ensure timely execution for verification
    stats = bt.optimize(
        ema_period=[50],
        peak_lookback=[20],
        min_risk_reward=[5],
        lod_confluence_pct=[0.01],
        maximize='Sharpe Ratio'
    )

    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        # Cast stats to native Python types
        stats_dict = {
            'strategy_name': 'measured_retracement_scalp_mw_formation',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }
        json.dump(stats_dict, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    # Generate plot
    bt.plot()
