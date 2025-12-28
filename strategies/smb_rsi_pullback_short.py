
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def sanitize_stats_for_json(stats):
    """
    Sanitizes the backtesting stats object to be JSON serializable.
    Removes non-serializable objects and converts numpy types to Python native types.
    """
    if stats is None:
        return {}

    # If stats is a Series, convert to dict
    if isinstance(stats, pd.Series):
        stats = stats.to_dict()

    # Create a copy to avoid modifying the original object
    sanitized = {}

    for key, value in stats.items():
        # Skip internal objects
        if isinstance(key, str) and key.startswith('_'):
            continue

        # Attempt to convert to a JSON-friendly format
        try:
            if pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (int, float, bool, str, type(None))):
                sanitized[key] = value
            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            else:
                # Convert numpy types to python types
                sanitized[key] = float(value)
        except (TypeError, ValueError):
            # If conversion fails, skip the key
            continue

    # Manually ensure required keys are present
    result = {
        'strategy_name': 'smb_rsi_pullback_short',
        'return': sanitized.get('Return [%]'),
        'sharpe': sanitized.get('Sharpe Ratio'),
        'max_drawdown': sanitized.get('Max. Drawdown [%]'),
        'win_rate': sanitized.get('Win Rate [%]'),
        'total_trades': sanitized.get('# Trades', 0)
    }
    return result

class SmbRsiPullbackShort(Strategy):
    """
    A strategy that shorts overextended stocks with very high RSI values,
    looking for a mean-reversion pullback.
    """
    # Optimizable parameters
    rsi_period = 14
    rsi_overbought = 90
    sma_period = 200
    atr_period = 14
    sl_atr_multiplier = 2.0
    tp_atr_multiplier = 4.0

    def init(self):
        """
        Initialize indicators.
        """
        # Convert data to numpy arrays for talib
        self.close = self.data.Close.astype(float)
        self.high = self.data.High.astype(float)
        self.low = self.data.Low.astype(float)

        # Initialize indicators using talib
        self.rsi = self.I(talib.RSI, self.close, timeperiod=self.rsi_period)
        self.sma = self.I(talib.SMA, self.close, timeperiod=self.sma_period)
        self.atr = self.I(talib.ATR, self.high, self.low, self.close, timeperiod=self.atr_period)

    def next(self):
        """
        Define the strategy logic for each candlestick.
        """
        # Context: We are in a strong uptrend (price is above long-term SMA)
        is_uptrend = self.close[-1] > self.sma[-1]

        # Condition: RSI is in overbought territory
        is_overbought = self.rsi[-1] > self.rsi_overbought

        # Trigger: A "failed continuation" candle (i.e., a bearish candle)
        is_failed_continuation = self.close[-1] < self.data.Open[-1]

        # If we are not in a position and all conditions are met, place a short order
        if not self.position and is_uptrend and is_overbought and is_failed_continuation:

            # Calculate Stop Loss and Take Profit
            stop_loss = self.high[-1] + self.atr[-1] * self.sl_atr_multiplier
            take_profit = self.close[-1] - self.atr[-1] * self.tp_atr_multiplier

            # Ensure take_profit is valid (below entry price for a short)
            if take_profit < self.close[-1]:
                self.sell(sl=stop_loss, tp=take_profit)

if __name__ == '__main__':
    import os
    import json

    # --- Configuration ---
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data/BTC-USD-15m.csv')
    INITIAL_CASH = 10000
    COMMISSION_RATE = 0.002 # 0.2%

    # --- Data Loading and Preparation ---
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    # A robust way to clean column names: strip whitespace and capitalize
    data.columns = [c.strip().capitalize() for c in data.columns]

    # --- Backtest Execution ---
    bt = Backtest(
        data,
        SmbRsiPullbackShort,
        cash=INITIAL_CASH,
        commission=COMMISSION_RATE
    )

    print("Running backtest with default parameters...")
    stats = bt.run()

    print("\nBacktest Results:")
    print(stats)

    # --- Save Results ---
    os.makedirs('results', exist_ok=True)
    RESULTS_PATH = 'results/temp_result.json'

    # Sanitize and save stats
    json_stats = sanitize_stats_for_json(stats)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(json_stats, f, indent=4)

    print(f"\nResults saved to {RESULTS_PATH}")

    # --- Plotting ---
    PLOT_FILENAME = 'results/smb_rsi_pullback_short.html'
    try:
        bt.plot(filename=PLOT_FILENAME, open_browser=False)
        print(f"Plot saved to {PLOT_FILENAME}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
