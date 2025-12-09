
import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

def generate_synthetic_data():
    # Generate a more realistic 24-hour forex-style dataset
    n_days = 90
    freq = '15min'
    periods = n_days * 24 * 4  # 90 days of 15-minute data

    # Create a date range
    timestamps = pd.date_range(start='2023-01-01', periods=periods, freq=freq, tz='UTC')

    # Base price and volatility
    base_price = 1.1000
    volatility = 0.0003

    # Generate random walks for price
    price_changes = np.random.normal(loc=0, scale=volatility, size=periods)
    price = base_price + np.cumsum(price_changes)

    # Create DataFrame
    data = pd.DataFrame(index=timestamps)
    data['Open'] = price
    data['High'] = data['Open'] + np.random.uniform(0, volatility * 2, size=periods)
    data['Low'] = data['Open'] - np.random.uniform(0, volatility * 2, size=periods)
    data['Close'] = data['Open'] + np.random.normal(0, volatility, size=periods)

    # Ensure High is the max and Low is the min
    data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)

    # Basic volume simulation
    data['Volume'] = np.random.randint(100, 5000, size=periods)

    return data

def preprocess_data(df):
    df = df.copy()

    # Define session times in UTC
    asia_start_hour = 0  # 00:00 UTC
    asia_end_hour = 8    # 08:00 UTC

    # Create a session identifier
    df['date'] = df.index.date
    df['hour'] = df.index.hour

    # Identify Asia session bars
    is_asia = (df['hour'] >= asia_start_hour) & (df['hour'] < asia_end_hour)

    # Calculate Asia session High and Low
    asia_session_data = df[is_asia].groupby('date').agg(
        Asia_High=('High', 'max'),
        Asia_Low=('Low', 'min')
    )

    # Map Asia session data to the entire day
    df = pd.merge(df, asia_session_data, left_on='date', right_index=True, how='left')

    # Forward fill the session levels to make them available throughout the day
    df[['Asia_High', 'Asia_Low']] = df[['Asia_High', 'Asia_Low']].ffill()

    df.dropna(inplace=True)
    return df

class SessionLiquidityGrabReversalStrategy(Strategy):
    asia_range_max_pct = 2.0  # Max Asia range in percentage

    def init(self):
        # State tracking variables to enforce one trade per day
        self.in_trade_today = False
        self.current_day = None

    def next(self):
        # Need at least 2 bars for pattern recognition
        if len(self.data.Close) < 2:
            return

        # Reset daily tracking variable at the start of a new day
        if self.data.index[-1].date() != self.current_day:
            self.current_day = self.data.index[-1].date()
            self.in_trade_today = False

        # If we already traded today or are in a position, do nothing
        if self.in_trade_today or self.position:
            return

        # Define trading session times (e.g., UK/US sessions)
        current_hour = self.data.index[-1].hour
        is_trading_session = current_hour >= 8  # After Asia session

        if not is_trading_session:
            return

        # --- Get pre-calculated session data ---
        asia_high = self.data.Asia_High[-1]
        asia_low = self.data.Asia_Low[-1]

        # Ensure we have valid session data for the current bar
        if pd.isna(asia_high) or pd.isna(asia_low):
            return

        # --- Strategy Filters ---
        # 1. Asia Range Filter: Only trade if the range is small enough
        asia_range = (asia_high - asia_low) / asia_low * 100
        if asia_range > self.asia_range_max_pct:
            return

        # --- Candlestick Pattern Recognition for the current candle [-1] ---

        # Bearish Engulfing: current bearish candle engulfs previous bullish candle
        is_bearish_engulfing = (self.data.Close[-1] < self.data.Open[-1] and
                                self.data.Open[-2] < self.data.Close[-2] and
                                self.data.Open[-1] >= self.data.Close[-2] and
                                self.data.Close[-1] < self.data.Open[-2])

        # Bullish Engulfing: current bullish candle engulfs previous bearish candle
        is_bullish_engulfing = (self.data.Close[-1] > self.data.Open[-1] and
                                self.data.Open[-2] > self.data.Close[-2] and
                                self.data.Open[-1] <= self.data.Close[-2] and
                                self.data.Close[-1] > self.data.Open[-2])

        # Pin Bar detection for the current candle [-1]
        body_size = abs(self.data.Open[-1] - self.data.Close[-1])
        total_range = self.data.High[-1] - self.data.Low[-1]

        is_bearish_pin_bar = False
        is_bullish_pin_bar = False

        if total_range > 0:
            upper_wick = self.data.High[-1] - max(self.data.Open[-1], self.data.Close[-1])
            lower_wick = min(self.data.Open[-1], self.data.Close[-1]) - self.data.Low[-1]

            # Bearish Pin Bar (Shooting Star): long upper wick, small body at the bottom
            is_bearish_pin_bar = (body_size / total_range < 0.33 and
                                  upper_wick / total_range > 0.6 and
                                  lower_wick / total_range < 0.2)

            # Bullish Pin Bar (Hammer): long lower wick, small body at the top
            is_bullish_pin_bar = (body_size / total_range < 0.33 and
                                  lower_wick / total_range > 0.6 and
                                  upper_wick / total_range < 0.2)

        is_bearish_reversal = is_bearish_engulfing or is_bearish_pin_bar
        is_bullish_reversal = is_bullish_engulfing or is_bullish_pin_bar

        # --- Stateless Entry Logic ---

        # SHORT ENTRY: Spike candle [-2] breaks Asia High, followed by a bearish reversal candle [-1]
        spike_up = self.data.High[-2] > asia_high and self.data.Close[-2] < asia_high
        if spike_up and is_bearish_reversal:
            # SL is the high of the entire 2-candle formation
            stop_loss = max(self.data.High[-1], self.data.High[-2])
            take_profit = asia_low

            # Final check before placing trade
            if self.data.Close[-1] > take_profit:
                 self.sell(sl=stop_loss, tp=take_profit)
                 self.in_trade_today = True

        # LONG ENTRY: Spike candle [-2] breaks Asia Low, followed by a bullish reversal candle [-1]
        spike_down = self.data.Low[-2] < asia_low and self.data.Close[-2] > asia_low
        if spike_down and is_bullish_reversal:
            # SL is the low of the entire 2-candle formation
            stop_loss = min(self.data.Low[-1], self.data.Low[-2])
            take_profit = asia_high

            # Final check before placing trade
            if self.data.Close[-1] < take_profit:
                self.buy(sl=stop_loss, tp=take_profit)
                self.in_trade_today = True

if __name__ == '__main__':
    # Generate and preprocess data
    data = generate_synthetic_data()
    data = preprocess_data(data)

    # Ensure data is not empty after preprocessing
    if data.empty:
        raise ValueError("Data is empty after preprocessing. Check session logic and date ranges.")

    # Initialize Backtest
    bt = Backtest(data, SessionLiquidityGrabReversalStrategy, cash=100_000, commission=.002)

    # Optimize the strategy
    stats = bt.optimize(
        asia_range_max_pct=np.arange(1.0, 3.1, 0.5).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.asia_range_max_pct > 0
    )

    print("Best stats:", stats)

    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)

    # Save results to JSON
    # Handle cases where no trades are made or Sharpe Ratio is NaN
    win_rate = stats.get('Win Rate [%]', 0)
    sharpe = stats.get('Sharpe Ratio', 0)

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'session_liquidity_grab_reversal',
            'return': float(stats.get('Return [%]', 0)),
            'sharpe': float(sharpe) if np.isfinite(sharpe) else None,
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)),
            'win_rate': float(win_rate) if np.isfinite(win_rate) else None,
            'total_trades': int(stats.get('# Trades', 0))
        }, f, indent=2)

    print("Results saved to results/temp_result.json")

    # Generate and save the plot, avoiding opening a browser in a headless environment
    plot_filename = 'results/session_liquidity_grab_reversal.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")
