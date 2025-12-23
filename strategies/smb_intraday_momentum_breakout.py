
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy
from backtesting.lib import crossover

class SMBIntradayMomentumBreakout(Strategy):
    """
    Implements an intraday momentum breakout strategy that buys when the price
    breaks above a short-term resistance level with strong momentum, using an
    ATR-based take profit and a trailing stop-loss.
    """
    # --- Strategy Parameters ---
    resistance_lookback = 30  # Lookback period for identifying resistance
    atr_period = 14           # ATR calculation period
    atr_multiplier = 3.0      # Multiplier for ATR-based take-profit
    trailing_sl_pct = 0.05    # 5% trailing stop-loss

    def init(self):
        """
        Initialize indicators and state variables.
        """
        # Calculate ATR for take-profit and risk management
        self.atr = self.I(ta.atr, pd.Series(self.data.Close), length=self.atr_period)

        # As a proxy for "relative strength", we'll use a long-term moving average
        self.sma_long = self.I(ta.sma, pd.Series(self.data.Close), length=200)

        # State variables for multi-stage exit logic
        self.initial_tp_price = None
        self.initial_tp_hit = False

    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        price = self.data.Close[-1]

        # --- Position Management ---
        if self.position:
            # On the first bar after entry, calculate and set the initial take-profit price
            if self.initial_tp_price is None:
                self.initial_tp_price = self.trades[0].entry_price + (self.atr[-1] * self.atr_multiplier)

            # 1. Partial Take-Profit: Close 50% of the position if the initial TP is hit
            if not self.initial_tp_hit and price >= self.initial_tp_price:
                self.position.close(portion=0.5)
                self.initial_tp_hit = True

            # 2. Trailing Stop-Loss: Trail the remaining position
            # The trailing SL is only active after the first TP is hit.
            if self.initial_tp_hit:
                new_sl = price * (1 - self.trailing_sl_pct)
                # Ensure the stop-loss only moves up, not down
                if self.trades[0].sl < new_sl:
                    self.trades[0].sl = new_sl
            return

        # --- Entry Conditions ---
        # Reset state variables when no position is open
        self.initial_tp_price = None
        self.initial_tp_hit = False

        # 1. Identify short-term resistance level
        resistance_level = self.data.High[-self.resistance_lookback:-1].max()

        # 2. Check for breakout and momentum
        is_breakout = price > resistance_level
        has_momentum = price > self.sma_long[-1] # Proxy for relative strength

        if is_breakout and has_momentum:
            # 3. Define stop-loss against the previous 5-minute higher low
            stop_loss = self.data.Low[-5:].min()

            # 4. Place buy order
            self.buy(sl=stop_loss)

# --- Backtesting Setup ---
if __name__ == '__main__':
    import os
    import json
    from backtesting import Backtest

    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Clean and rename columns to match backtesting.py conventions
    data.columns = [col.strip().capitalize() for col in data.columns]

    bt = Backtest(data, SMBIntradayMomentumBreakout, cash=100_000, commission=.002)

    print("Running backtest...")
    stats = bt.run()

    print("\nBacktest Stats:")
    print(stats)

    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Save stats to a JSON file
    results_path = 'results/temp_result.json'
    with open(results_path, 'w') as f:
        json.dump(stats.to_dict(), f, indent=4)
    print(f"\nResults saved to {results_path}")

    # Generate and save the plot
    plot_path = 'results/smb_intraday_momentum_breakout.html'
    try:
        bt.plot(filename=plot_path)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
