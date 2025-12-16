import pandas as pd
from backtesting import Backtest, Strategy
import pandas_ta as ta
import json
import os

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it's JSON-serializable.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            sanitized[key] = None
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif pd.isna(value):
            sanitized[key] = None
        else:
            try:
                sanitized[key] = float(value)
            except (ValueError, TypeError):
                sanitized[key] = str(value)
    return sanitized

class WeinsteinStage4ShortSell(Strategy):
    """
    Implements the Weinstein Stage 4 Short Sell strategy.

    NOTE: The Relative Strength Line, a key component of Weinstein's original
    strategy, has been omitted from this implementation. A true RSL requires
    comparing the asset's performance to a broad market index (e.g., S&P 500),
    and this backtest environment does not provide external market data.
    The logic is therefore based on the price action, volume, and moving average
    criteria described.
    """
    # Strategy parameters
    ma_period = 30  # 30-week MA
    support_lookback = 52  # 52 weeks for support zone
    rr_ratio = 2.0  # Risk-reward ratio
    volume_ma_period = 20 # Lookback for volume MA

    def init(self):
        """
        Initialize indicators and preprocess data.
        """
        # --- Weekly Data Calculation ---
        # Resample the 15-minute data to weekly
        df_weekly = self.data.df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # Calculate 30-week SMA on weekly data
        df_weekly['sma30w'] = ta.sma(df_weekly['Close'], length=self.ma_period)

        # Calculate 52-week support level (lowest low of the lookback period)
        df_weekly['support'] = df_weekly['Low'].rolling(self.support_lookback, min_periods=1).min()

        # --- Map Weekly Indicators to 15-min Data ---
        # Create a temporary DataFrame to hold the weekly values
        weekly_indicators = df_weekly[['sma30w', 'support']].copy()

        # Reindex to the 15-minute timeframe and forward-fill
        # This effectively maps the weekly value to all 15-min bars within that week
        mapped_indicators = weekly_indicators.reindex(self.data.df.index, method='ffill')

        # --- Add Indicators to the Main Data Object for Plotting and Access ---
        # The self.I() function is used to register indicators with the framework
        self.sma30w = self.I(lambda x: x, mapped_indicators['sma30w'].values, name="SMA30w")
        self.support = self.I(lambda x: x, mapped_indicators['support'].values, name="Support")

        # Calculate Volume MA on the original 15-minute data
        self.volume_sma = self.I(ta.sma, pd.Series(self.data.Volume), self.volume_ma_period, name="VolumeSMA")

        # --- Strategy State Variables ---
        self.stage4_confirmed = False
        self.breakdown_level = 0

    def next(self):
        """
        Implements the trading logic for each bar.
        """
        # Ensure we have enough data
        if len(self.data.Close) < 5 or pd.isna(self.sma30w[-1]) or pd.isna(self.support[-1]):
            return

        # --- CONDITION 1: Identify Stage 4 and Breakdown ---
        is_sma_declining = self.sma30w[-1] < self.sma30w[-5]  # Check over 5 bars for a clear trend
        price_below_sma = self.data.Close[-1] < self.sma30w[-1]

        if not self.stage4_confirmed:
            # A breakdown occurs when price closes below the long-term support
            if is_sma_declining and price_below_sma and self.data.Close[-1] < self.support[-2]:
                # Confirm with heavy volume
                if self.data.Volume[-1] > self.volume_sma[-1]:
                    self.stage4_confirmed = True
                    self.breakdown_level = self.support[-2] # The support level that was broken
                    # Optional: Log the breakdown event
                    # print(f"{self.data.index[-1]}: Stage 4 Breakdown Confirmed. Level: {self.breakdown_level}")
            return # Wait for the next bar after confirming breakdown

        # --- CONDITION 2: Wait for Pullback and Entry ---
        if self.stage4_confirmed and not self.position:
            # Pullback condition: price rallies back up towards the breakdown level
            is_pullback = self.data.High[-1] >= self.breakdown_level

            # Entry condition: pullback occurs on light volume and shows reversal
            if is_pullback:
                # Check for light volume during the pullback attempt
                is_light_volume = self.data.Volume[-1] < self.volume_sma[-1]

                # Check for a bearish reversal candle (e.g., closes red)
                is_reversal_candle = self.data.Close[-1] < self.data.Open[-1]

                if is_light_volume and is_reversal_candle:
                    # --- EXECUTE SHORT TRADE ---
                    entry_price = self.data.Close[-1]

                    # Place Stop-Loss above the recent pullback high (resistance)
                    stop_loss = self.data.High[-1] * 1.01 # Adding a small buffer

                    # Calculate Take-Profit based on Risk-Reward Ratio
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * self.rr_ratio)

                    # Ensure SL and TP are valid before placing trade
                    if entry_price < stop_loss and take_profit > 0:
                        self.sell(sl=stop_loss, tp=take_profit)
                        # print(f"{self.data.index[-1]}: SELL ORDER PLACED at {entry_price}")

        # --- EXIT / INVALIDATION LOGIC ---
        # If the Stage 4 downtrend is invalidated, reset the state
        if self.stage4_confirmed and self.data.Close[-1] > self.sma30w[-1]:
            self.stage4_confirmed = False
            self.breakdown_level = 0
            # Optional: Log the invalidation event
            # print(f"{self.data.index[-1]}: Stage 4 Invalidated. Price crossed above 30W SMA.")

if __name__ == '__main__':
    # The user-provided data path was `data/BTC-USD-15m.csv`, but the file is
    # located in a `crypto` subdirectory.
    data_path = 'data/crypto/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # The source CSV is malformed: the header has leading spaces and a trailing
    # comma, which causes pandas to misinterpret column names and create an
    # extra empty column. We correct this by explicitly selecting the first five
    # data columns and assigning the required names for `backtesting.py`.
    data = data.iloc[:, :5]
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    bt = Backtest(data, WeinsteinStage4ShortSell, cash=100_000, commission=.002)

    stats = bt.run()

    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Sanitize the stats for JSON serialization
    sanitized_stats = sanitize_stats(stats)

    # Save the results
    result_data = {
        'strategy_name': 'weinstein_stage4_short_sell',
        'return': sanitized_stats.get('Return [%]'),
        'sharpe': sanitized_stats.get('Sharpe Ratio'),
        'max_drawdown': sanitized_stats.get('Max. Drawdown [%]'),
        'win_rate': sanitized_stats.get('Win Rate [%]'),
        'total_trades': sanitized_stats.get('# Trades')
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_data, f, indent=2)
        f.write('\\n')

    print("Backtest results saved to results/temp_result.json")

    # Generate and save the plot
    try:
        plot_filename = 'results/weinstein_stage4_short_sell.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
