import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# --- Indicator wrappers for backtesting.py ---

def PtaMACD(close_series, fast=12, slow=26, signal=9):
    """
    Wrapper for pandas_ta.macd to return a tuple of numpy arrays (MACD, Hist, Signal)
    as expected by backtesting.py's self.I().
    """
    # Note: pandas_ta returns MACD, Histogram, Signal
    macd = ta.macd(close=pd.Series(close_series), fast=fast, slow=slow, signal=signal, append=False)
    return macd.iloc[:, 0].values, macd.iloc[:, 1].values, macd.iloc[:, 2].values

def PtaMFI(high_series, low_series, close_series, volume_series, length=14):
    """
    Wrapper for pandas_ta.mfi to return a numpy array
    as expected by backtesting.py's self.I().
    """
    mfi = ta.mfi(
        high=pd.Series(high_series),
        low=pd.Series(low_series),
        close=pd.Series(close_series),
        volume=pd.Series(volume_series),
        length=length,
        append=False
    )
    return mfi.values

def PtaATR(high_series, low_series, close_series, length=14):
    """
    Wrapper for pandas_ta.atr to return a numpy array
    as expected by backtesting.py's self.I().
    """
    atr = ta.atr(
        high=pd.Series(high_series),
        low=pd.Series(low_series),
        close=pd.Series(close_series),
        length=length,
        append=False
    )
    return atr.values

# --- Strategy Definition ---

class MarketCipherProxyStrategy(Strategy):
    """
    A proxy for a "Market Cipher" style strategy.

    This strategy combines MACD for trend direction and MFI for identifying
    overbought/oversold conditions to generate trading signals. ATR is used
    for dynamic stop-loss and take-profit placement.

    Long Entry:
    - MACD is above the MACD Signal line (trend is up).
    - MFI crosses below the oversold level (pullback/entry point).

    Short Entry:
    - MACD is below the MACD Signal line (trend is down).
    - MFI crosses above the overbought level (reversal point).
    """
    # --- MACD Parameters ---
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # --- MFI Parameters ---
    mfi_period = 14
    mfi_overbought = 80
    mfi_oversold = 20

    # --- Risk Management Parameters ---
    atr_period = 14
    sl_atr_multiplier = 1.5
    rr_ratio = 2.0  # Risk:Reward Ratio

    def init(self):
        """
        Initialize indicators.
        """
        # MACD
        self.macd, self.macd_hist, self.macd_signal_line = self.I(
            PtaMACD,
            self.data.Close,
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal
        )

        # MFI
        self.mfi = self.I(
            PtaMFI,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume,
            length=self.mfi_period
        )

        # ATR for Stop-Loss
        self.atr = self.I(
            PtaATR,
            self.data.High,
            self.data.Low,
            self.data.Close,
            length=self.atr_period
        )

    def next(self):
        """
        Define the trading logic on each bar.
        """
        price = self.data.Close[-1]

        # If a position is already open, do nothing.
        if self.position:
            return

        # Get the latest ATR value for SL/TP calculation.
        # Skip if ATR is zero or NaN to avoid invalid calculations.
        atr_value = self.atr[-1]
        if not atr_value or pd.isna(atr_value) or atr_value == 0:
            return

        # --- Long Entry Conditions ---
        # 1. Trend confirmation: MACD line is above the signal line.
        # 2. Momentum confirmation: MFI has crossed up from the oversold zone.
        is_bullish_trend = self.macd[-1] > self.macd_signal_line[-1]
        is_oversold_exit = crossover(self.mfi, self.mfi_oversold)

        if is_bullish_trend and is_oversold_exit:
            # Calculate Stop-Loss and Take-Profit
            stop_loss = price - atr_value * self.sl_atr_multiplier
            take_profit = price + (price - stop_loss) * self.rr_ratio

            # Place buy order with calculated SL/TP
            # Add a guard to ensure SL/TP are valid relative to the price
            if stop_loss < price and take_profit > price:
                self.buy(sl=stop_loss, tp=take_profit)
            return

        # --- Short Entry Conditions ---
        # 1. Trend confirmation: MACD line is below the signal line.
        # 2. Momentum confirmation: MFI has crossed down from the overbought zone.
        is_bearish_trend = self.macd[-1] < self.macd_signal_line[-1]
        is_overbought_exit = crossover(self.mfi_overbought, self.mfi)

        if is_bearish_trend and is_overbought_exit:
            # Calculate Stop-Loss and Take-Profit
            stop_loss = price + atr_value * self.sl_atr_multiplier
            take_profit = price - (stop_loss - price) * self.rr_ratio

            # Place sell order with calculated SL/TP
            # Add a guard to ensure SL/TP are valid relative to the price
            if stop_loss > price and take_profit < price:
                self.sell(sl=stop_loss, tp=take_profit)
            return

# --- Backtesting Execution ---

def generate_synthetic_data(n_points=2500):
    """Generates synthetic data for testing the strategy."""
    print("Generating synthetic data...")
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))

    # Create a plausible price series with trends and volatility
    price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.2)
    price += np.sin(np.linspace(0, 150, n_points)) * 3 # Add some cyclicality
    price = price.rolling(window=10).mean().fillna(method='bfill') # Smooth it out

    # Ensure Open, High, Low, Close are consistent
    open_price = price.shift(1).fillna(method='bfill')
    high_price = pd.concat([open_price, price], axis=1).max(axis=1) + np.random.uniform(0, 0.5, n_points)
    low_price = pd.concat([open_price, price], axis=1).min(axis=1) - np.random.uniform(0, 0.5, n_points)

    data = pd.DataFrame({
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': price,
        'Volume': np.random.randint(100, 5000, n_points)
    }, index=index)
    return data

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to make it JSON serializable.
    Removes non-serializable types like DataFrames and handles numpy/pandas types.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            continue
        if isinstance(value, (np.floating, np.integer)):
            sanitized[key] = float(value) if np.isfinite(value) else None
        elif isinstance(value, int):
            sanitized[key] = int(value)
        elif isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        elif pd.isna(value) or value is pd.NA:
            sanitized[key] = None
        elif key.startswith('_'): # Don't serialize private attributes
            continue
        else:
            sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        try:
            # More robustly load the data, handling potential header issues
            data = pd.read_csv(
                data_path,
                index_col='datetime',
                parse_dates=True,
                header=0,
                names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unnamed'],
                usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
            # Sanitize column names just in case there are extra spaces
            data.columns = [col.strip().capitalize() for col in data.columns]
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'.")
        data = generate_synthetic_data()

    # Instantiate the Backtest
    bt = Backtest(data, MarketCipherProxyStrategy, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Sanitize and save stats to JSON
    final_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    # Generate and save the plot
    try:
        plot_filename = f"results/strategy_e75ea19009fd.html"
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
