from backtesting import Strategy
import pandas as pd
import pandas_ta as ta

def TDI(close, rsi_length=13, bb_length=34, bb_std=1.6185, rsi_signal_length=7, rsi_ma_length=2, rsi_ma_type='sma'):
    """
    Custom indicator function to calculate the Traders Dynamic Index (TDI).
    This implementation calculates the components manually.
    """
    close_series = pd.Series(close)
    rsi = ta.rsi(close=close_series, length=rsi_length)

    # Calculate Volatility Bands (Bollinger Bands on RSI)
    bbands = ta.bbands(close=rsi, length=bb_length, std=bb_std)
    upper_band_col = [col for col in bbands.columns if col.startswith('BBU_')][0]
    lower_band_col = [col for col in bbands.columns if col.startswith('BBL_')][0]
    upper_band = bbands[upper_band_col]
    lower_band = bbands[lower_band_col]

    # Calculate Signal Line and Market Baseline (Moving Averages of RSI)
    signal_line = ta.sma(rsi, length=rsi_signal_length)
    market_baseline = ta.sma(rsi, length=bb_length) # The middle band of the BB is the market baseline

    return rsi.values, signal_line.values, market_baseline.values, upper_band.values, lower_band.values

class TdiSharkFinStrategy(Strategy):
    """
    Implements the TDI Shark Fin entry pattern strategy.
    This strategy identifies reversal opportunities by looking for the RSI line to break
    out of the TDI's volatility bands and then hook back, forming a "shark fin".
    """
    # TDI parameters
    rsi_length = 13
    rsi_signal_length = 7
    bb_length = 34
    bb_std = 1.6185

    # Risk management parameters
    atr_period = 14
    atr_multiplier = 2.0

    def init(self):
        """
        Initialize the strategy's indicators.
        """
        close = self.data.Close
        # Initialize the TDI indicator
        self.rsi, self.signal_line, self.market_baseline, self.upper_band, self.lower_band = self.I(
            TDI,
            close,
            rsi_length=self.rsi_length,
            bb_length=self.bb_length,
            bb_std=self.bb_std,
            rsi_signal_length=self.rsi_signal_length
        )

        # Initialize ATR for stop loss calculation
        self.atr = self.I(lambda x, n: ta.atr(high=pd.Series(self.data.High), low=pd.Series(self.data.Low), close=pd.Series(x), length=n).values,
                          close, self.atr_period)

        # State variables to track the shark fin pattern
        self.short_setup_active = False
        self.long_setup_active = False

        # State variables for two-step take profit
        self.long_tp_setup = False
        self.short_tp_setup = False

    def next(self):
        """
        Main strategy logic for entry and exit.
        """
        # --- Short Entry Logic ---
        # A short setup is active if the RSI breaks above the upper volatility band.
        if self.rsi[-1] > self.upper_band[-1]:
            self.short_setup_active = True
            self.long_setup_active = False # Invalidate any long setup

        # If a short setup is active, check for the entry trigger: RSI crossing back below the signal line.
        if self.short_setup_active and self.rsi[-1] < self.signal_line[-1] and not self.position:
            # Place short order
            sl = self.data.High[-1] + self.atr[-1] * self.atr_multiplier
            self.sell(sl=sl, size=0.1)
            # Reset setup state after entry
            self.short_setup_active = False

        # --- Long Entry Logic ---
        # A long setup is active if the RSI breaks below the lower volatility band.
        if self.rsi[-1] < self.lower_band[-1]:
            self.long_setup_active = True
            self.short_setup_active = False # Invalidate any short setup

        # If a long setup is active, check for the entry trigger: RSI crossing back above the signal line.
        if self.long_setup_active and self.rsi[-1] > self.signal_line[-1] and not self.position:
            # Place long order
            sl = self.data.Low[-1] - self.atr[-1] * self.atr_multiplier
            self.buy(sl=sl, size=0.1)
            # Reset setup state after entry
            self.long_setup_active = False

        # --- Exit Logic (Take Profit) ---
        if self.position.is_long:
            # Step 1: Check if RSI has reached the opposite (upper) band
            if self.rsi[-1] > self.upper_band[-1]:
                self.long_tp_setup = True

            # Step 2: If setup is active, check for the exit trigger (RSI crosses back below signal line)
            if self.long_tp_setup and self.rsi[-1] < self.signal_line[-1]:
                self.position.close()
                self.long_tp_setup = False # Reset state after closing

        elif self.position.is_short:
            # Step 1: Check if RSI has reached the opposite (lower) band
            if self.rsi[-1] < self.lower_band[-1]:
                self.short_tp_setup = True

            # Step 2: If setup is active, check for the exit trigger (RSI crosses back above signal line)
            if self.short_tp_setup and self.rsi[-1] > self.signal_line[-1]:
                self.position.close()
                self.short_tp_setup = False # Reset state after closing

if __name__ == '__main__':
    import os
    import json
    from backtesting import Backtest
    from backtesting.lib import FractionalBacktest

    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    strategy_name = TdiSharkFinStrategy.__name__
    output_dir = 'results'

    # --- Data Loading and Preparation ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True, skipinitialspace=True)
    # Clean column names: strip whitespace, title case, and remove unnamed columns
    data.columns = [c.strip().title() for c in data.columns]
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # --- Backtesting ---
    bt = FractionalBacktest(data, TdiSharkFinStrategy, cash=10_000, commission=.002)
    stats = bt.run()

    # --- Reporting ---
    print(stats)

    # --- Save Results ---
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    plot_filename = os.path.join(output_dir, f'{strategy_name}.html')
    bt.plot(filename=plot_filename, open_browser=False)

    # Save stats to JSON
    stats_dict = dict(stats)

    # Sanitize stats for JSON serialization
    for key, value in list(stats_dict.items()):
        if isinstance(value, pd.DataFrame):
            stats_dict.pop(key)
        elif isinstance(value, pd.Timestamp):
            stats_dict[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            stats_dict[key] = str(value)
        elif pd.isna(value):
            stats_dict[key] = None
        elif isinstance(value, (float, int)):
            # Basic types are fine
            continue
        else: # Catch other potential non-serializable types like internal objects
            stats_dict.pop(key, None)

    json_filename = os.path.join(output_dir, 'temp_result.json')
    with open(json_filename, 'w') as f:
        json.dump(stats_dict, f, indent=4)

    print(f"\nBacktest results saved to {output_dir}/")
    print(f"- Stats: {json_filename}")
    print(f"- Plot: {plot_filename}")
