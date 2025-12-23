
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# It's recommended to use talib or pandas_ta for indicators
# For VWAP, we can calculate it manually or use a library
# Let's use pandas_ta for this example
import pandas_ta as ta

from enum import Enum

class TradeState(Enum):
    WAITING_FOR_SETUP = 1
    MONITORING_FOR_ENTRY = 2
    POSITION_OPEN = 3

class SmbIpoDay3Reversion(Strategy):
    """
    Strategy to fade the exhaustion of a strong 3-day momentum move in recent IPOs.
    This implementation adapts the logic for a continuous market like BTC-USD.
    """

    # --- Strategy Parameters ---
    day1_min_gain_pct = 2.0
    day2_min_gain_pct = 1.0
    day3_gap_up_pct = 0.5
    sl_buffer_pct = 0.5

    def init(self):
        """
        Initialize the strategy.
        """
        self.state = TradeState.WAITING_FOR_SETUP
        self.day_3_high = None
        self.day_1_high = None

        # Make pre-calculated data available to the strategy
        self.is_day_3 = self.I(lambda x: x, self.data.is_day_3)
        self.vwap = self.I(lambda x: x, self.data.vwap)
        self.vwap_3d = self.I(lambda x: x, self.data.vwap_3d)
        self.day_3_high_series = self.I(lambda x: x, self.data.day_3_high)
        self.day_1_high_series = self.I(lambda x: x, self.data.day_1_high)


    def next(self):
        """
        The main strategy logic that is executed on each bar.
        """
        price = self.data.Close[-1]

        # --- State Machine Logic ---
        if self.state == TradeState.WAITING_FOR_SETUP:
            if self.is_day_3[-1]:
                self.day_3_high = self.day_3_high_series[-1]
                self.day_1_high = self.day_1_high_series[-1]
                self.state = TradeState.MONITORING_FOR_ENTRY

        elif self.state == TradeState.MONITORING_FOR_ENTRY:
            # Invalidate setup if a new day starts
            if self.data.index[-1].date() != self.data.index[-2].date():
                self.state = TradeState.WAITING_FOR_SETUP
                return

            # Entry condition: Clear momentum shift (close below VWAP) on a bearish candle
            if price < self.vwap[-1] and self.data.Close[-1] < self.data.Open[-1]:

                # --- Risk Management ---
                stop_loss = self.day_3_high * (1 + self.sl_buffer_pct / 100)

                # --- Take Profit Levels ---
                # Target 1: 3-day VWAP
                # Target 2: Day 1 High
                tp1 = self.vwap_3d[-1]
                tp2 = self.day_1_high

                # Ensure targets are valid for a short
                if price > tp1 and price > tp2:
                    # Enter one trade; exits will be managed manually.
                    # The SL is set here, but TPs are handled in the POSITION_OPEN state.
                    self.sell(sl=stop_loss)
                    self.state = TradeState.POSITION_OPEN

        elif self.state == TradeState.POSITION_OPEN:
            # --- Manual Exit Logic for Multi-TP ---
            # Position has two take profit levels
            tp1 = self.vwap_3d[-1]
            tp2 = self.day_1_high

            # Close half the position if TP1 is hit
            if price < tp1 and self.position.size > 0.5:
                self.position.close(portion=0.5)

            # Check for TP2 on the remaining position
            if price < tp2 and self.position.size > 0:
                 self.position.close()

            # If position is closed, reset state
            if not self.position:
                self.state = TradeState.WAITING_FOR_SETUP


def preprocess_data(df: pd.DataFrame):
    """
    Pre-processes the 15-minute data to add signals needed for the strategy.
    - Identifies Day 1, 2, 3 momentum pattern.
    - Calculates daily and multi-day VWAP.
    - Maps daily levels (Day 1 High, Day 3 High) to the 15m timeframe.
    """
    if 'datetime' not in df.columns:
        df['datetime'] = df.index
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # --- Daily Data Aggregation ---
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    daily_df.dropna(inplace=True)

    # --- Identify Momentum Days ---
    daily_df['gain'] = daily_df['close'].pct_change() * 100

    # Day 1: Strong momentum
    daily_df['is_day_1'] = daily_df['gain'] > SmbIpoDay3Reversion.day1_min_gain_pct

    # Day 2: Continued momentum (takes out Day 1 high)
    daily_df['day_1_high_raw'] = daily_df['high'].shift(1)
    daily_df['is_day_2'] = (daily_df['gain'] > SmbIpoDay3Reversion.day2_min_gain_pct) & \
                           (daily_df['high'] > daily_df['day_1_high_raw']) & \
                           (daily_df['is_day_1'].shift(1))

    # Day 3: Gap up and potential exhaustion
    daily_df['gap_up'] = (daily_df['open'] - daily_df['close'].shift(1)) / daily_df['close'].shift(1) * 100
    daily_df['is_day_3_raw'] = (daily_df['gap_up'] > SmbIpoDay3Reversion.day3_gap_up_pct) & \
                             (daily_df['is_day_2'].shift(1))

    # Carry forward the Day 1 high to the Day 3 row for merging
    daily_df['day_1_high'] = daily_df['day_1_high_raw'].shift(1)
    daily_df['day_3_high'] = daily_df['high']

    # --- Map Daily Signals back to 15m DataFrame using .map for stability ---
    day_3_signals = daily_df[daily_df['is_day_3_raw']]

    # Create dictionaries to map dates to signal values
    day_1_high_map = day_3_signals['day_1_high'].to_dict()
    day_3_high_map = day_3_signals['day_3_high'].to_dict()

    # Map the signals to the intraday dataframe's date
    df['day_1_high'] = df.index.to_series().dt.date.map(day_1_high_map)
    df['day_3_high'] = df.index.to_series().dt.date.map(day_3_high_map)

    # A "day 3" is any bar where the mapped Day 1 high is not null
    df['is_day_3'] = ~df['day_1_high'].isnull()

    # --- Calculate Intraday and Multi-day VWAP ---
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df.ta.vwap(append=True) # Calculates intraday VWAP, resets daily
    df.rename(columns={'VWAP_D': 'vwap'}, inplace=True)

    # Calculate 3-day rolling VWAP
    bars_in_3_days = 96 * 3
    typ_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['vwap_3d'] = (typ_price * df['Volume']).rolling(bars_in_3_days).sum() / df['Volume'].rolling(bars_in_3_days).sum()

    # --- Data Cleaning ---
    # Fill NaN values for signals to avoid issues during backtest
    # We use bfill then ffill to handle NaNs at the beginning and end
    df[['day_1_high', 'day_3_high']] = df[['day_1_high', 'day_3_high']].bfill().ffill()
    df.dropna(subset=['vwap', 'vwap_3d'], inplace=True)

    return df


if __name__ == '__main__':
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv')
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure the data file is in the correct location.")
        # As a fallback, try to generate some synthetic data for demonstration
        from backtesting.test import GOOG
        df = GOOG.copy()
        df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }, inplace=True)
        # For synthetic data, we need a datetime index
        df.index.name = 'datetime'
        df.reset_index(inplace=True)

    # Standardize column names to lowercase and strip whitespace
    df.columns = [x.strip().lower() for x in df.columns]

    # The CSV has a trailing comma, creating an extra column. Select only what's needed.
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    # --- Pre-process Data ---
    data = preprocess_data(df)

    # --- Run Backtest ---
    bt = Backtest(data, SmbIpoDay3Reversion, cash=100_000, commission=.002)

    stats = bt.run()
    print(stats)

    # --- Plotting ---
    try:
        bt.plot(filename='results/smb_ipo_day_3_reversion.html', open_browser=False)
    except Exception as e:
        print(f"Error plotting: {e}")

    # --- Save Results ---
    import json

    results = {
        'strategy_name': 'smb_ipo_day_3_reversion',
        'return': stats['Return [%]'],
        'sharpe': stats['Sharpe Ratio'],
        'max_drawdown': stats['Max. Drawdown [%]'],
        'win_rate': stats['Win Rate [%]'],
        'total_trades': stats['# Trades'],
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Backtest complete. Results saved to results/temp_result.json and plot saved to results/smb_ipo_day_3_reversion.html")
