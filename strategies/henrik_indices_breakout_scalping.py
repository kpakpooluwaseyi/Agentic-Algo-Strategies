from backtesting import Backtest, Strategy
import pandas as pd
import pandas_ta as ta
import json

def get_hour(series):
    """Pass-through function to register the hour series."""
    return series

class HenrikIndicesBreakoutScalpingStrategy(Strategy):
    """
    Implements a breakout scalping strategy based on identifying a consolidation
    range and trading a high-momentum breakout, with aggressive break-even
    stop management.
    """
    # --- Optimizable Parameters ---
    breakout_window = 24      # Lookback period to define the trading range (e.g., 24 bars)
    atr_multiplier = 1.5      # Multiplier for ATR to confirm a "nervous" breakout candle
    rr_ratio = 1.5            # Risk-to-Reward ratio for take-profit
    session_start_hour = 14   # Session start time in UTC (e.g., 14:00 for US open)
    session_end_hour = 21     # Session end time in UTC (e.g., 21:00 for US close)
    breakeven_profit_pct = 0.005 # Profit % to trigger move to break-even (e.g., 0.5%)
    sl_buffer_pct = 0.001     # Buffer % for stop-loss placement (e.g., 0.1%)

    def init(self):
        """Initialize the strategy's state and indicators."""
        # --- Indicators ---
        # Calculate ATR using pandas_ta and register it.
        # The input arrays must be converted to pandas Series for pandas_ta to work correctly.
        self.atr = self.I(lambda: ta.atr(
            high=pd.Series(self.data.High),
            low=pd.Series(self.data.Low),
            close=pd.Series(self.data.Close),
            length=14
        ).values, name='ATR')

        # Register the pre-calculated hour column as an indicator for session filtering.
        self.hour = self.I(get_hour, self.data.df['hour'], name='hour')

        # --- State Management ---
        # Tracks if the break-even stop has been set for the current trade.
        self.breakeven_triggered = False

    def next(self):
        """The main strategy logic executed on each bar."""
        current_hour = self.hour[-1]

        # --- Reset break-even trigger if not in a position ---
        if not self.position:
            self.breakeven_triggered = False

        # --- Session Filter ---
        if not (self.session_start_hour <= current_hour <= self.session_end_hour):
            return

        # --- Break-Even Logic ---
        if self.position and not self.breakeven_triggered:
            if self.position.is_long and self.data.High[-1] >= self.position.entry_price * (1 + self.breakeven_profit_pct):
                self.trades[0].sl = self.position.entry_price
                self.breakeven_triggered = True
            elif self.position.is_short and self.data.Low[-1] <= self.position.entry_price * (1 - self.breakeven_profit_pct):
                self.trades[0].sl = self.position.entry_price
                self.breakeven_triggered = True

        # --- Entry Logic ---
        if not self.position:
            lookback_data = self.data.df.iloc[-self.breakout_window:]
            range_high = lookback_data['High'].max()
            range_low = lookback_data['Low'].min()

            current_close = self.data.Close[-1]
            current_atr = self.atr[-1]
            candle_body = abs(self.data.Close[-1] - self.data.Open[-1])

            # Bullish Breakout
            if current_close > range_high and candle_body > current_atr * self.atr_multiplier:
                sl = range_high * (1 - self.sl_buffer_pct)
                tp = current_close + (current_close - sl) * self.rr_ratio
                self.buy(sl=sl, tp=tp)

            # Bearish Breakout
            elif current_close < range_low and candle_body > current_atr * self.atr_multiplier:
                sl = range_low * (1 + self.sl_buffer_pct)
                tp = current_close - (sl - current_close) * self.rr_ratio
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    # --- Load Data ---
    # Using 'data/BTC-USD-15m.csv' as requested
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Error: 'data/BTC-USD-15m.csv' not found. Please ensure the data file is in the correct directory.")
        # As a fallback, generate some synthetic data to allow the script to run
        from backtesting.test import GOOG
        data = GOOG.copy()
        data.index = pd.to_datetime(data.index)


    # --- Data Pre-processing ---
    # Clean and capitalize column names to match backtesting.py requirements
    data.columns = [col.strip().capitalize() for col in data.columns]

    # Add the 'hour' column required for the session filter
    data['hour'] = data.index.hour

    # --- Backtest ---
    # Use FractionalBacktest to handle high-priced assets like BTC
    from backtesting.lib import FractionalBacktest
    bt = FractionalBacktest(data, HenrikIndicesBreakoutScalpingStrategy, cash=100_000, commission=.002)

    stats = bt.run()

    print("--- Backtest Results ---")
    print(stats)
    print("----------------------")

    # --- Results ---
    import os

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Save results to JSON
    # Sanitize stats for JSON serialization
    stats_dict = dict(stats)
    # Remove non-serializable objects
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    for key, value in list(stats_dict.items()):
        if pd.isna(value):
            stats_dict[key] = None
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict.pop(key) # Timestamps and Timedeltas are not needed in the JSON output

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)

    # Generate plot
    try:
        plot_filename = 'results/henrik_indices_breakout_scalping.html'
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
