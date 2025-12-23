import pandas as pd
from backtesting import Backtest, Strategy
import json
import os

# --- Indicator Functions ---

def fvg(high: pd.Series, low: pd.Series):
    """
    Detects Fair Value Gaps (FVG).
    A bullish FVG is when the low of the third candle is above the high of the first candle.
    A bearish FVG is when the high of the third candle is below the low of the first candle.
    """
    high = pd.Series(high)
    low = pd.Series(low)
    # Bullish FVG: Low of candle [i] is higher than the High of candle [i-2]
    bullish_fvg_upper = high.shift(2)
    bullish_fvg_lower = low.shift(0)
    bullish_fvg_condition = bullish_fvg_lower > bullish_fvg_upper

    # Bearish FVG: High of candle [i] is lower than the Low of candle [i-2]
    bearish_fvg_upper = high.shift(0)
    bearish_fvg_lower = low.shift(2)
    bearish_fvg_condition = bearish_fvg_upper < bearish_fvg_lower

    fvg_upper = pd.Series(index=high.index, dtype=float)
    fvg_lower = pd.Series(index=high.index, dtype=float)
    fvg_type = pd.Series(index=high.index, dtype=float) # 1 for bullish, -1 for bearish

    fvg_upper[bullish_fvg_condition] = bullish_fvg_lower[bullish_fvg_condition]
    fvg_lower[bullish_fvg_condition] = bullish_fvg_upper[bullish_fvg_condition]
    fvg_type[bullish_fvg_condition] = 1

    fvg_upper[bearish_fvg_condition] = bearish_fvg_lower[bearish_fvg_condition]
    fvg_lower[bearish_fvg_condition] = bearish_fvg_upper[bearish_fvg_condition]
    fvg_type[bearish_fvg_condition] = -1

    return fvg_type, fvg_upper, fvg_lower

# --- Strategy Definition ---

class ExpansionCandleTrading(Strategy):
    """
    Strategy based on identifying a stop run, displacement with a Fair Value Gap (FVG),
    and entering on a retracement into that FVG.
    """
    risk_reward_ratio = 2.0
    sl_buffer_pct = 0.01
    invalidation_period = 5 # Bars to wait for entry before invalidating setup

    def init(self):
        self.fvg_type, self.fvg_upper, self.fvg_lower = self.I(
            fvg, self.data.High, self.data.Low, name="FVG"
        )
        # State machine variables
        self.setup_direction = 0  # 1 for long, -1 for short
        self.setup_bar_index = 0
        self.entry_fvg_upper = None
        self.entry_fvg_lower = None
        self.entry_sl = None
        self.entry_tp = None

    def next(self):
        # --- Invalidation Logic ---
        # If a setup is active and it's been too long, reset
        if self.setup_direction != 0 and (len(self.data.Close) - 1 - self.setup_bar_index > self.invalidation_period):
            self.reset_setup()

        # --- Position Management ---
        # Only check for entries if we don't have an open position
        if self.position:
            return

        # --- Entry Logic ---
        # If we are in a "waiting for long" state
        if self.setup_direction == 1 and self.entry_sl is not None:
            # If price retraces into the FVG
            if self.data.Low[-1] <= self.entry_fvg_upper:
                self.buy(sl=self.entry_sl, tp=self.entry_tp)
                self.reset_setup()
        # If we are in a "waiting for short" state
        elif self.setup_direction == -1 and self.entry_sl is not None:
            # If price retraces into the FVG
            if self.data.High[-1] >= self.entry_fvg_lower:
                self.sell(sl=self.entry_sl, tp=self.entry_tp)
                self.reset_setup()

        # --- Setup Detection (if no setup is active and no position) ---
        if self.setup_direction == 0 and not self.position and len(self.data.Close) > 3:
            # Bearish Setup: Stop run up, then displacement down creating a bearish FVG
            is_stop_run_up = self.data.High[-2] > self.data.High[-3]
            is_displacement_down_with_fvg = self.fvg_type[-1] == -1

            if is_stop_run_up and is_displacement_down_with_fvg:
                self.setup_direction = -1
                self.setup_bar_index = len(self.data.Close) - 1
                self.entry_fvg_upper = self.fvg_upper[-1]
                self.entry_fvg_lower = self.fvg_lower[-1]

                stop_loss = self.data.High[-2] * (1 + self.sl_buffer_pct)
                entry_price = self.entry_fvg_lower # Enter when price touches the bottom of the bearish FVG

                # Check for valid risk before setting state
                if stop_loss > entry_price:
                    take_profit = entry_price - (stop_loss - entry_price) * self.risk_reward_ratio
                    if take_profit < entry_price:
                        self.entry_sl = stop_loss
                        self.entry_tp = take_profit

            # Bullish Setup: Stop run down, then displacement up creating a bullish FVG
            is_stop_run_down = self.data.Low[-2] < self.data.Low[-3]
            is_displacement_up_with_fvg = self.fvg_type[-1] == 1

            if is_stop_run_down and is_displacement_up_with_fvg:
                self.setup_direction = 1
                self.setup_bar_index = len(self.data.Close) - 1
                self.entry_fvg_upper = self.fvg_upper[-1]
                self.entry_fvg_lower = self.fvg_lower[-1]

                stop_loss = self.data.Low[-2] * (1 - self.sl_buffer_pct)
                entry_price = self.entry_fvg_upper # Enter when price touches the top of the bullish FVG

                # Check for valid risk before setting state
                if stop_loss < entry_price:
                    take_profit = entry_price + (entry_price - stop_loss) * self.risk_reward_ratio
                    if take_profit > entry_price:
                        self.entry_sl = stop_loss
                        self.entry_tp = take_profit

    def reset_setup(self):
        self.setup_direction = 0
        self.setup_bar_index = 0
        self.entry_fvg_upper = None
        self.entry_fvg_lower = None
        self.entry_sl = None
        self.entry_tp = None

# --- Main Execution ---
if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

    # Drop the unnamed column if it exists (from trailing comma in header)
    if 'Unnamed: 6' in df.columns:
        df.drop('Unnamed: 6', axis=1, inplace=True)

    df.dropna(inplace=True)

    # Ensure columns are correctly named
    df.columns = [c.strip().capitalize() for c in df.columns]

    bt = Backtest(df, ExpansionCandleTrading, cash=100_000, commission=.002)

    stats = bt.run()
    print(stats)

    # --- Save results ---
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Sanitize stats for JSON output
    sanitized_stats = {key: str(value) if isinstance(value, (pd.Timestamp, pd.Timedelta)) else value
                       for key, value in stats.to_dict().items() if not key.startswith('_')}

    with open(os.path.join(results_dir, 'temp_result.json'), 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    # --- Plot results ---
    plot_filename = os.path.join(results_dir, 'expansion_candle_trading.html')
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
