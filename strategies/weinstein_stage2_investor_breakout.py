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


class WeinsteinStage2InvestorBreakout(Strategy):
    """
    Implements a simplified version of the Weinstein Stage 2 Investor Breakout strategy.
    This adaptation uses a long-term SMA to identify stages and looks for breakouts
    on high volume after a basing period using weekly chart analysis.
    """
    # --- Strategy Parameters ---
    # Note: Lookback now applies to 15m data, but logic is driven by weekly signals
    resistance_lookback = 96 * 250 # Approx. 1 year of 15m bars (for Stage 1 base)
    volume_factor = 1.5           # Volume must be 1.5x the average
    ma_slope_flat_threshold = 0.05 # Weekly MA slope threshold for Stage 1/3
    pullback_tolerance = 0.03     # 3% tolerance for pullback entry

    def init(self):
        """
        Initialize state variables. Indicators are pre-calculated.
        """
        # --- State Machine Variables ---
        self.stage = 1
        self.resistance_level = None
        self.support_level = None
        self.breakout_price = None

        # Adjust lookback if data is shorter than the lookback period (for synthetic data)
        actual_lookback = min(self.resistance_lookback, len(self.data.Close) - 1)

        # Use volume MA on the 15m data for breakout confirmation
        self.volume_ma = self.I(ta.sma, pd.Series(self.data.Volume), length=actual_lookback)

    def next(self):
        """
        Main strategy logic executed on each 15-minute bar.
        """
        # Use the pre-calculated weekly slope
        current_slope = self.data.SMA30_Slope[-1]
        current_price = self.data.Close[-1]

        # --- Stage 1: Basing ---
        if self.stage == 1:
            if abs(current_slope) < self.ma_slope_flat_threshold:
                self.resistance_level = self.data.High[-self.resistance_lookback:].max()
                self.support_level = self.data.Low[-self.resistance_lookback:].min() # Define support for SL

                volume_surge = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_factor
                price_breakout = current_price > self.resistance_level
                ma_turning_up = current_slope > 0
                price_above_ma = current_price > self.data.SMA30[-1]

                # RS must be > 1 (strong) and ideally rising
                rs_is_strong = self.data.RS[-1] > 1
                rs_is_trending = self.data.RS[-1] > self.data.RS[-5] # Check 5-period RS trend on 15m data

                if not self.position and price_breakout and volume_surge and ma_turning_up and price_above_ma and rs_is_strong and rs_is_trending:
                    # Place buy order with a stop-loss at the bottom of the Stage 1 base
                    self.buy(size=0.5, sl=self.support_level)
                    self.breakout_price = current_price
                    self.stage = 2
            return

        # --- Stage 2: Advancing ---
        if self.stage == 2:
            if self.position.size > 0 and self.position.size < 1.0:
                 pullback_cond = abs(current_price - self.breakout_price) / self.breakout_price < self.pullback_tolerance
                 volume_contract = self.data.Volume[-1] < self.volume_ma[-1]
                 if pullback_cond and volume_contract:
                     self.buy(size=0.5)

            if self.position and current_slope < self.ma_slope_flat_threshold:
                self.stage = 3
                self.support_level = self.data.Low[-self.resistance_lookback:].min()
                if self.position.size == 1.0:
                    self.position.close(portion=0.5)
                elif self.position.size > 0:
                    self.position.close()

        # --- Stage 3: Topping ---
        if self.stage == 3:
            if self.position and current_slope < 0 and current_price < self.support_level:
                self.stage = 4
                self.position.close()

        # --- Stage 4: Declining ---
        if self.stage == 4:
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

    bt = Backtest(data, WeinsteinStage2InvestorBreakout, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    print("Backtest Stats:")
    print(stats)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Save stats to JSON
    results = {
        'strategy_name': 'weinstein_stage2_investor_breakout',
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
        plot_filename = 'results/weinstein_stage2_investor_breakout.html'
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
