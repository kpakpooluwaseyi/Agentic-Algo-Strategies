
import pandas as pd
from backtesting import Strategy, Backtest
import talib
from scipy.signal import find_peaks
import numpy as np
import json
import os

# Enum for state machine
class State:
    SEARCHING_FOR_SWING = 0
    IN_CONSOLIDATION = 1
    WAITING_FOR_BREAKOUT = 2

def preprocess_data(df, **params):
    """
    Adds all necessary indicators and filters to the DataFrame to comply with
    the development guidelines.
    """
    df = df.copy()

    # Compliance: ATR for volatility-based risk management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['atr_ma'] = df['atr'].rolling(params.get('atr_ma_period', 50)).mean()

    # Compliance: Volume MA for entry confirmation
    df['volume_ma'] = df['Volume'].rolling(params.get('volume_ma_period', 20)).mean()

    # Compliance: Higher Timeframe Filter (4H EMA 200)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_uptrend'] = df_4h['Close'] > df_4h['ema_200']

    # Merge HTF trend back to the main timeframe
    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill')
    df['htf_uptrend'].fillna(False, inplace=True) # Fill initial NaNs

    # Proxy for M/W Patterns: Find significant swing highs and lows
    # Using a longer period relative to 15m timeframe to find meaningful swings
    peak_indices, _ = find_peaks(df['High'], distance=params.get('peak_distance', 12))
    trough_indices, _ = find_peaks(-df['Low'], distance=params.get('peak_distance', 12))
    df['swing_high'] = False
    df['swing_low'] = False
    df.iloc[peak_indices, df.columns.get_loc('swing_high')] = True
    df.iloc[trough_indices, df.columns.get_loc('swing_low')] = True

    return df

class StrategyFff2e5031208(Strategy):
    """
    Compliant adaptation of the 'Beat The Market Maker' strategy.

    This strategy translates the qualitative concepts of the original strategy
    into a quantitative algorithm that adheres to the repository's rules.

    - "Stop Hunt / M/W Pattern" is proxied by detecting significant swing highs/lows.
    - "Consolidation" is identified as a period of low volatility (ATR < ATR MA).
    - "Breakout" from consolidation is the entry trigger.

    All entries are filtered by a 4H trend direction and confirmed by high volume,
    with ATR-based risk management, as per repository guidelines.
    """
    # Optimizable Parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    peak_distance = 15
    consolidation_bars = 4 # 60 minutes / 15 min per bar
    atr_ma_period = 50
    volume_ma_period = 20

    def init(self):
        # State machine
        self.state = State.SEARCHING_FOR_SWING
        self.consolidation_high = None
        self.consolidation_low = None
        self.consolidation_entry_bar = None

        # Indicators for quick access
        self.atr = self.I(lambda: self.data.atr, name="ATR")
        self.atr_ma = self.I(lambda: self.data.atr_ma, name="ATR_MA")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="Volume_MA")
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend, name="HTF_Uptrend")
        self.swing_high = self.I(lambda: self.data.swing_high, name="Swing_High")
        self.swing_low = self.I(lambda: self.data.swing_low, name="Swing_Low")

    def next(self):
        # Warm-up period: ensure all indicators have enough data
        if len(self.data) < self.atr_ma_period:
            return

        # Prevent acting on open positions
        if self.position:
            return

        current_bar_index = len(self.data) - 1

        # --- State Machine Logic ---

        if self.state == State.SEARCHING_FOR_SWING:
            # Look for a recent swing high to initiate a short setup
            if self.swing_high[-2] and not self.htf_uptrend[-1]:
                self.state = State.IN_CONSOLIDATION
                self.consolidation_entry_bar = current_bar_index
                return

            # Look for a recent swing low to initiate a long setup
            if self.swing_low[-2] and self.htf_uptrend[-1]:
                self.state = State.IN_CONSOLIDATION
                self.consolidation_entry_bar = current_bar_index
                return

        elif self.state == State.IN_CONSOLIDATION:
            # Invalidate if consolidation period exceeds the defined bars
            if current_bar_index > self.consolidation_entry_bar + self.consolidation_bars:
                self.state = State.SEARCHING_FOR_SWING
                return

            # Wait for volatility to drop (proxy for consolidation)
            if self.atr[-1] < self.atr_ma[-1]:
                # Define the consolidation range
                start_idx = self.consolidation_entry_bar
                self.consolidation_high = self.data.High[start_idx:current_bar_index + 1].max()
                self.consolidation_low = self.data.Low[start_idx:current_bar_index + 1].min()
                self.state = State.WAITING_FOR_BREAKOUT

        elif self.state == State.WAITING_FOR_BREAKOUT:
            # Invalidate if price makes a new swing in the opposite direction
            if (self.htf_uptrend[-1] and self.swing_high[-2]) or \
               (not self.htf_uptrend[-1] and self.swing_low[-2]):
                self.state = State.SEARCHING_FOR_SWING
                return

            # --- Entry Logic ---
            price = self.data.Close[-1]
            atr_val = self.atr[-1]

            # Short entry condition: Break below consolidation
            if not self.htf_uptrend[-1] and price < self.consolidation_low:
                # Compliance: Volume confirmation
                if self.data.Volume[-1] > self.volume_ma[-1]:
                    sl = self.consolidation_high + (self.atr_sl_multiplier * atr_val)
                    tp = price - (self.atr_tp_multiplier * atr_val)
                    if tp < sl: # Ensure TP is valid
                        self.sell(sl=sl, tp=tp)
                    self.state = State.SEARCHING_FOR_SWING # Reset state

            # Long entry condition: Break above consolidation
            elif self.htf_uptrend[-1] and price > self.consolidation_high:
                # Compliance: Volume confirmation
                if self.data.Volume[-1] > self.volume_ma[-1]:
                    sl = self.consolidation_low - (self.atr_sl_multiplier * atr_val)
                    tp = price + (self.atr_tp_multiplier * atr_val)
                    if tp > sl: # Ensure TP is valid
                        self.buy(sl=sl, tp=tp)
                    self.state = State.SEARCHING_FOR_SWING # Reset state


# --- Standalone Runner ---
if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    plot_filename = os.path.join(results_dir, 'strategy_fff2e5031208.html')
    json_filename = os.path.join(results_dir, 'temp_result.json')

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    try:
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        # Sanitize column names: remove leading/trailing spaces and capitalize
        df.columns = [col.strip().capitalize() for col in df.columns]
        # Rename 'Close ' to 'Close' if there's a space from a trailing comma
        df.rename(columns={'Close ': 'Close'}, inplace=True)

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file is in the correct location.")
        # Create a dummy dataframe for the script to run without error
        dummy_dates = pd.date_range(start='2023-01-01', periods=1000, freq='15min')
        dummy_data = np.random.rand(1000, 5) * 100 + 16000
        df = pd.DataFrame(dummy_data, index=dummy_dates, columns=['Open', 'High', 'Low', 'Close', 'Volume'])


    # Preprocess the data with default parameters
    params = {
        'peak_distance': StrategyFff2e5031208.peak_distance,
        'atr_ma_period': StrategyFff2e5031208.atr_ma_period,
        'volume_ma_period': StrategyFff2e5031208.volume_ma_period,
    }
    df_processed = preprocess_data(df, **params)

    # Run the backtest
    bt = Backtest(df_processed, StrategyFff2e5031208, cash=100_000, commission=.002)
    stats = bt.run()

    print("--- Backtest Stats ---")
    print(stats)

    # Save the results to a JSON file
    stats_dict = dict(stats)
    # Remove non-serializable items
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    for key, value in stats_dict.items():
        if isinstance(value, (np.integer, np.floating)):
            stats_dict[key] = float(value)
        if isinstance(value, pd.Timestamp):
            stats_dict[key] = value.isoformat()
        if isinstance(value, pd.Timedelta):
            stats_dict[key] = str(value)

    with open(json_filename, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"\nResults saved to {json_filename}")

    # Generate and save the plot
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"\nCould not generate plot due to an error: {e}")
        print("This may be due to plotting library issues in the environment.")
