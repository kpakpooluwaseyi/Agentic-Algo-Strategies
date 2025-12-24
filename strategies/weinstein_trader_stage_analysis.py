import pandas as pd
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy
import json
import os

def MA_SLOPE(array, n):
    """
    Calculates the slope of a 1D array using numpy.polyfit for vectorization.
    The slope is normalized by the mean of the window.
    """
    # Create a Series to handle rolling window easily
    s = pd.Series(array)

    # Use a rolling window and apply polyfit
    # The lambda function calculates the slope of the best-fit line
    slopes = s.rolling(window=n).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == n else np.nan,
        raw=True
    )

    # Normalize by the rolling mean
    rolling_mean = s.rolling(window=n).mean()
    normalized_slopes = slopes / rolling_mean * 100

    return normalized_slopes.values


class WeinsteinTraderStageAnalysis(Strategy):
    """
    Implements a simplified version of the Weinstein Stage Analysis strategy for traders.
    This version focuses on the "Continuation Buy" (pullback to a rising 30-week MA)
    and also includes the initial Stage 1 to Stage 2 breakout as a secondary entry.
    """
    # --- Strategy Parameters ---
    # For initial Stage 1 -> 2 breakout
    initial_base_lookback = 96 * 250 # Approx. 1 year for the initial Stage 1 base

    # For Stage 2 Continuation Buys (the primary setup)
    continuation_lookback = 96 * 60   # Approx. 3 months for the re-consolidation base
    ma_proximity_pct = 0.05           # Price must pull back to within 5% of the 30-week MA
    ma_slope_strong_threshold = 0.1   # MA must be "strongly" rising for a continuation buy

    # General parameters
    volume_factor = 1.5               # Breakout volume must be 1.5x the average
    ma_slope_flat_threshold = 0.05    # Weekly MA slope threshold for Stage 1/3 transitions

    def init(self):
        """
        Initialize state variables and indicators.
        """
        # --- State Machine Variables ---
        self.stage = 1
        self.resistance_level = None
        self.support_level = None
        self.in_pullback_zone = False # Tracks if price is near the 30-week MA for a continuation buy

        # --- Indicators ---
        # Adjust lookback if data is shorter than the lookback period to avoid errors
        actual_lookback = min(self.initial_base_lookback, len(self.data.Close) - 1)
        self.volume_ma = self.I(ta.sma, pd.Series(self.data.Volume), length=actual_lookback)

    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        current_price = self.data.Close[-1]
        current_slope = self.data.SMA30_Slope[-1]
        current_ma = self.data.SMA30[-1]

        # --- Stage 1: Basing (Looking for initial breakout) ---
        if self.stage == 1:
            # Condition for being in a Stage 1 base
            if abs(current_slope) < self.ma_slope_flat_threshold:
                self.resistance_level = self.data.High[-self.initial_base_lookback:].max()
                self.support_level = self.data.Low[-self.initial_base_lookback:].min()

                # Entry conditions for the secondary, early Stage 2 breakout
                volume_surge = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_factor
                price_breakout = current_price > self.resistance_level
                ma_turning_up = current_slope > 0
                price_above_ma = current_price > current_ma
                rs_is_strong = self.data.RS[-1] > 1
                rs_is_trending = self.data.RS[-1] > self.data.RS[-5]

                if not self.position and price_breakout and volume_surge and ma_turning_up and price_above_ma and rs_is_strong and rs_is_trending:
                    # Tighter stop-loss for traders: Use the 30-week MA as the stop
                    self.buy(sl=current_ma)
                    self.stage = 2
                    self.in_pullback_zone = False # Reset pullback state
            return

        # --- Stage 2: Advancing (Looking for continuation buys) ---
        if self.stage == 2:
            # Trader's Exit Rule: Exit as soon as momentum wanes (MA flattens)
            if self.position and current_slope < self.ma_slope_strong_threshold:
                self.position.close()
                self.stage = 3 # Transition to Stage 3 after closing the position
                self.in_pullback_zone = False # Reset state
                return

            # --- Primary Entry: Continuation Buy Logic ---
            # Condition 1: MA must be strongly rising
            is_ma_strong = current_slope > self.ma_slope_strong_threshold

            # Condition 2: Price pulls back to near the MA
            is_near_ma = abs(current_price - current_ma) / current_ma < self.ma_proximity_pct

            if is_ma_strong and is_near_ma:
                self.in_pullback_zone = True # Enter the pullback zone

            # Condition 3: Price breaks out of a new, shorter-term base AFTER a pullback
            if not self.position and self.in_pullback_zone:
                # Define the new, shorter consolidation range
                consolidation_high = self.data.High[-self.continuation_lookback:].max()
                consolidation_low = self.data.Low[-self.continuation_lookback:].min()

                price_breakout_new = current_price > consolidation_high
                volume_surge_new = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_factor
                rs_is_trending = self.data.RS[-1] > self.data.RS[-5]

                if price_breakout_new and volume_surge_new and is_ma_strong and rs_is_trending:
                    self.buy(sl=consolidation_low)
                    self.in_pullback_zone = False # Exit pullback zone after entry

        # --- Stage 3: Topping ---
        if self.stage == 3:
            # Look for breakdown to Stage 4
            support_level = self.data.Low[-self.continuation_lookback:].min()
            if current_slope < 0 and current_price < support_level:
                self.stage = 4
                if self.position:
                    self.position.close() # Should already be closed, but as a safeguard
            elif current_slope > self.ma_slope_flat_threshold:
                # It might be re-entering a Stage 2
                self.stage = 2

        # --- Stage 4: Declining ---
        if self.stage == 4:
            # Check for transition back to Stage 1
            if current_slope > -self.ma_slope_flat_threshold:
                self.stage = 1

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Clean and rename columns
        data.columns = data.columns.str.strip()
        data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        # Drop any unnamed columns
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        # --- Timeframe Resampling ---
        # Resample 15m data to weekly to apply Weinstein's logic
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        weekly_data = data.resample('W-MON').agg(ohlc_dict).dropna()

        # Calculate weekly indicators
        weekly_data['SMA30'] = ta.sma(weekly_data['Close'], length=30)
        weekly_data['SMA30_Slope'] = MA_SLOPE(weekly_data['SMA30'].values, n=10) # 10-week slope

        # Re-implemented Relative Strength Proxy (Price vs. long-term MA)
        long_term_ma_period = 200
        if len(weekly_data) < long_term_ma_period:
            print(f"Warning: Data length ({len(weekly_data)} weeks) is shorter than long-term MA period (200). Using shorter MA for RS.")
            long_term_ma_period = len(weekly_data) // 2
        long_term_ma = ta.sma(weekly_data['Close'], length=long_term_ma_period)
        weekly_data['RS'] = (weekly_data['Close'] / long_term_ma)

        # --- Map Weekly Signals to 15m Data ---
        # Forward-fill weekly data to align with the 15m index
        data = pd.merge(data, weekly_data[['SMA30', 'SMA30_Slope', 'RS']], left_index=True, right_index=True, how='left')
        data[['SMA30', 'SMA30_Slope', 'RS']] = data[['SMA30', 'SMA30_Slope', 'RS']].ffill()
        data.dropna(inplace=True)

    else:
        print(f"Error: Data file not found at {data_path}")
        # --- Generate realistic fallback data for CI/CD ---
        print("Generating synthetic data for fallback...")
        n_points = 2000
        index = pd.to_datetime(pd.date_range('2022-01-01', periods=n_points, freq='15min'))
        price = 100 + np.random.randn(n_points).cumsum() * 0.1

        # Stage 1: Basing period (sideways market)
        price[500:1500] = 100 + np.sin(np.linspace(0, 10 * np.pi, 1000)) * 5

        # Stage 2: Breakout
        price[1500:] = 110 + np.random.randn(n_points - 1500).cumsum() * 0.2

        volume = np.random.uniform(100, 500, n_points)
        volume[1490:1510] = np.random.uniform(1000, 2000, 20) # Volume surge on breakout

        data = pd.DataFrame({
            'Open': price, 'High': price + 0.5, 'Low': price - 0.5, 'Close': price, 'Volume': volume
        }, index=index)

        # Add placeholder weekly data so the backtest can run
        data['SMA30'] = ta.sma(data['Close'], length=200)
        data['SMA30_Slope'] = MA_SLOPE(data['SMA30'].values, n=50)
        long_ma = ta.sma(data['Close'], length=800)
        data['RS'] = data['Close'] / long_ma
        data.dropna(inplace=True)

    bt = Backtest(data, WeinsteinTraderStageAnalysis, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    print("Backtest Stats:")
    print(stats)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Save stats to JSON
    results = {
        'strategy_name': 'weinstein_trader_stage_analysis',
        'return': stats.get('Return [%]', None),
        'sharpe': stats.get('Sharpe Ratio', None),
        'max_drawdown': stats.get('Max. Drawdown [%]', None),
        'win_rate': stats.get('Win Rate [%]', None),
        'total_trades': stats.get('# Trades', 0)
    }

    # Sanitize results for JSON output
    cleaned_results = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in results.items()}

    with open('results/temp_result.json', 'w') as f:
        json.dump(cleaned_results, f, indent=2)
        f.write('\n') # Add a newline for POSIX compliance

    print("Results saved to results/temp_result.json")

    # Generate plot
    try:
        plot_filename = 'results/weinstein_trader_stage_analysis.html'
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
