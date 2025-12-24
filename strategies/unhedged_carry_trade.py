import pandas as pd
import numpy as np
from backtesting import Strategy
import pandas_ta as ta
import json

def generate_synthetic_carry_data(n_points=1000):
    """
    Generates synthetic data simulating a currency pair's price and the
    interest rates for a "target" (high-yield) and "funding" (low-yield) currency.
    """
    # Create a datetime index
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_points, freq='D'))

    # 1. Generate Interest Rates
    # Funding rate: stable and low
    funding_rate = pd.Series(np.random.normal(1.0, 0.05, n_points), index=dates).cumsum() + 1
    funding_rate = pd.Series(funding_rate).rolling(window=50).mean() * 0.5 + 1.0 # Smooth and floor at 1%

    # Target rate: fluctuates more, with clear periods of being higher
    periods = 4
    cycles = np.sin(np.linspace(0, periods * np.pi, n_points)) * 2.5 # Cycles between -2.5 and +2.5
    noise = np.random.normal(0, 0.2, n_points)
    target_rate = funding_rate + cycles + noise + 2.0 # Ensure it's generally higher
    target_rate = pd.Series(target_rate).rolling(window=10).mean().fillna(3.0) # Smooth

    # 2. Generate Price Data influenced by the rate differential
    rate_spread = target_rate - funding_rate
    # Price drifts up when spread is positive, down when negative
    price_drift = rate_spread * 0.05
    price_noise = np.random.normal(0, 0.8, n_points)
    price = (price_drift + price_noise).cumsum() + 100

    # 3. Create OHLC data
    data = pd.DataFrame(index=dates)
    data['Close'] = price
    data['Open'] = data['Close'].shift(1).bfill()
    data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 0.5, n_points)
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 0.5, n_points)

    # 4. Add rates to the DataFrame
    data['Target_Rate'] = target_rate
    data['Funding_Rate'] = funding_rate

    # Ensure no NaN values remain
    data.bfill(inplace=True)
    data.ffill(inplace=True)

    # Add a Volume column for compatibility
    data['Volume'] = np.random.randint(100, 1000, size=n_points)

    return data

# Custom Indicator functions
def ATR(high, low, close, period):
    """Calculates the Average True Range using pandas_ta."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    return ta.atr(high=high_s, low=low_s, close=close_s, length=period)

class UnhedgedCarryTrade(Strategy):
    """
    Implements a true unhedged carry trade strategy based on interest rate
    differentials. It goes long when the target currency's interest rate
    is significantly higher than the funding currency's rate.
    """
    # --- Strategy Parameters ---
    # The minimum positive spread required to enter a trade
    entry_spread_threshold = 0.5

    # --- Risk Management Parameters ---
    risk_reward_ratio = 1.5
    atr_period = 14
    atr_multiplier = 2.0

    def init(self):
        """Initialize the indicators."""
        # Custom indicator to calculate the interest rate spread
        self.rate_spread = self.I(lambda: self.data.Target_Rate - self.data.Funding_Rate, name="RateSpread")
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        """Define the entry and exit logic."""
        # --- ENTRY LOGIC ---
        # The core of the carry trade: go long if the target rate is
        # sufficiently higher than the funding rate.
        if not self.position and self.rate_spread[-1] > self.entry_spread_threshold:
            atr_value = self.atr[-1]
            entry_price = self.data.Close[-1]

            # Set SL and TP for risk management
            sl = entry_price - atr_value * self.atr_multiplier
            tp = entry_price + (entry_price - sl) * self.risk_reward_ratio

            # Basic validation before placing the trade
            if tp > entry_price and sl < entry_price:
                self.buy(sl=sl, tp=tp)

        # --- EXIT LOGIC ---
        # Exit if the positive carry disappears (rate spread is no longer favorable).
        # The primary exit is the SL/TP; this is a secondary condition.
        elif self.position and self.rate_spread[-1] <= 0:
            self.position.close()

if __name__ == '__main__':
    from backtesting import Backtest
    import os

    # --- Data Generation ---
    # The strategy now uses synthetic data that includes interest rates.
    data = generate_synthetic_carry_data(n_points=2000)

    # --- Backtest Execution ---
    bt = Backtest(data, UnhedgedCarryTrade, cash=10_000, commission=.002)

    print("Running backtest with synthetic data...")
    stats = bt.run()
    print(stats)

    # --- Results Saving ---
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        """A robust way to sanitize stats for JSON serialization."""
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (np.integer, np.floating, int, float)):
                if not np.isfinite(value):
                    sanitized[key] = None
                else:
                    sanitized[key] = value
            elif isinstance(value, str):
                 sanitized[key] = value

        # Manually add strategy name for clarity
        sanitized['strategy_name'] = 'unhedged_carry_trade'
        return sanitized

    final_stats = sanitize_stats(stats)

    # Remove non-serializable objects from the final stats
    final_stats.pop('_strategy', None)
    final_stats.pop('_equity_curve', None)
    final_stats.pop('_trades', None)

    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=4)
    print("\nBacktest results saved to results/temp_result.json")

    # --- Plot Generation ---
    try:
        plot_filename = 'results/unhedged_carry_trade_synthetic.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
