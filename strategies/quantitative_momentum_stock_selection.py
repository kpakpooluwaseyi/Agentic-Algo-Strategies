import pandas as pd
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy
import json
import os

class QuantitativeMomentumStockSelection(Strategy):
    """
    Implements a simplified quantitative momentum strategy adapted for a single asset.
    It buys when both absolute (time-series) and relative (cross-sectional proxy)
    momentum are positive, and sells when either condition is no longer met.
    """
    # --- Strategy Parameters ---
    # Threshold for positive time-series momentum (e.g., > 0% return)
    ts_momentum_threshold = 0
    # Threshold for strong cross-sectional momentum (e.g., RS > 1)
    cs_momentum_threshold = 1.0

    def init(self):
        """
        Initializes the strategy. No custom indicators are needed as signals
        are pre-calculated and available directly on the data object.
        """
        pass

    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        # --- Get the latest momentum signals ---
        time_series_momentum = self.data.Time_Series_Momentum[-1]
        cross_sectional_momentum = self.data.RS_Proxy[-1]

        # --- Entry Conditions ---
        # Both momentum factors must be positive
        is_strong_momentum = (
            time_series_momentum > self.ts_momentum_threshold and
            cross_sectional_momentum > self.cs_momentum_threshold
        )

        # --- Exit Conditions ---
        # Either momentum factor is no longer positive
        is_weak_momentum = (
            time_series_momentum <= self.ts_momentum_threshold or
            cross_sectional_momentum <= self.cs_momentum_threshold
        )

        # --- Execute Trades ---
        if not self.position:
            if is_strong_momentum:
                # No SL/TP defined in the strategy, managed by holding during strong momentum
                self.buy()
        elif is_weak_momentum:
            self.position.close()


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

        # --- Timeframe Resampling for Long-Term Indicators ---
        ohlc_dict = {
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }
        weekly_data = data.resample('W-MON').agg(ohlc_dict).dropna()

        # --- Indicator Calculation ---
        # 1. Cross-Sectional Momentum Proxy (Relative Strength vs. own history)
        long_term_ma_period = 30 # 30-week SMA for long-term trend
        if len(weekly_data) < long_term_ma_period:
            long_term_ma_period = len(weekly_data) // 2 # Adapt for shorter datasets

        weekly_data['Long_Term_SMA'] = ta.sma(weekly_data['Close'], length=long_term_ma_period)
        weekly_data['RS_Proxy'] = weekly_data['Close'] / weekly_data['Long_Term_SMA']

        # 2. Time-Series Momentum (Absolute Momentum)
        # 52-week (approx. 12 months) rate of change
        time_series_momentum_period = 52
        if len(weekly_data) <= time_series_momentum_period:
            print(f"Warning: Data length ({len(weekly_data)} weeks) is too short for 52-week momentum. Adapting lookback.")
            time_series_momentum_period = len(weekly_data) - 1
        weekly_data['Time_Series_Momentum'] = weekly_data['Close'].pct_change(periods=time_series_momentum_period)

        # --- Map Weekly Signals to 15m Data ---
        data = pd.merge(data, weekly_data[['RS_Proxy', 'Time_Series_Momentum']], left_index=True, right_index=True, how='left')
        data[['RS_Proxy', 'Time_Series_Momentum']] = data[['RS_Proxy', 'Time_Series_Momentum']].ffill()
        data.dropna(inplace=True)

    else:
        print(f"Error: Data file not found at {data_path}")
        # --- Generate realistic fallback data for CI/CD ---
        print("Generating synthetic data for fallback...")
        n_points = 2000
        index = pd.to_datetime(pd.date_range('2022-01-01', periods=n_points, freq='15min'))
        price = 100 + np.random.randn(n_points).cumsum() * 0.1

        data = pd.DataFrame({
            'Open': price, 'High': price + 0.5, 'Low': price - 0.5, 'Close': price, 'Volume': np.random.uniform(100, 500, n_points)
        }, index=index)
        data.dropna(inplace=True)

    bt = Backtest(data, QuantitativeMomentumStockSelection, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    print("Backtest Stats:")
    print(stats)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Save stats to JSON
    results_dict = {
        'strategy_name': 'quantitative_momentum_stock_selection',
    }
    # Add all stats from the backtest run to the results dictionary
    for key, value in stats.items():
        results_dict[key] = value

    # Sanitize results for JSON output
    def sanitize_for_json(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items() if not k.startswith('_')}
        return obj

    cleaned_results = sanitize_for_json(results_dict)

    with open('results/temp_result.json', 'w') as f:
        json.dump(cleaned_results, f, indent=2)
        f.write('\n')

    print("Results saved to results/temp_result.json")

    # Generate plot
    try:
        plot_filename = 'results/quantitative_momentum_stock_selection.html'
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
