import numpy as np
import pandas as pd
from scipy.stats import norm
from backtesting import Strategy
from backtesting.lib import crossover

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price and delta for a European call option.

    S: Current stock price
    K: Option strike price
    T: Time to maturity in years
    r: Risk-free interest rate
    sigma: Volatility of the underlying asset
    """
    if T <= 0:
        return (max(0, S - K), 1 if S > K else 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)

    return price, delta

class BlackScholesDeltaHedging(Strategy):
    """
    Implements a dynamic delta hedging strategy for a short European call option.

    The strategy aims to maintain a delta-neutral position by continuously
    rebalancing its holding in the underlying asset to match the option's
    changing delta. This is a classic risk-management strategy for option
    sellers or market makers.

    On each bar, the strategy:
    1. Calculates the time remaining until the option's maturity.
    2. Computes the option's current delta using the Black-Scholes model.
    3. Adjusts its position in the underlying asset to match the new delta,
       buying or selling the difference required to rebalance the hedge.

    NOTE: The P&L of this backtest reflects the performance of the hedge ONLY.
    A perfect hedge would have a P&L that is the exact inverse of the P&L
    of the short option, resulting in a net-zero outcome.
    """

    # --- Strategy Parameters ---
    K = 17000      # Strike price of the option
    r = 0.05       # Annual risk-free interest rate (e.g., 5%)
    sigma = 0.85   # Annualized volatility of the underlying
    T_days = 30    # Time to maturity in days
    n_options = 10_000 # Number of options. Must be > 1 to ensure unit sizing.

    def init(self):
        # Pre-calculate time constants
        self.T_in_seconds = self.T_days * 24 * 60 * 60
        self.start_datetime = self.data.index[0]
        self.seconds_in_year = 365.25 * 24 * 60 * 60

    def next(self):
        # --- 1. Calculate Time to Maturity ---
        current_datetime = self.data.index[-1]
        seconds_elapsed = (current_datetime - self.start_datetime).total_seconds()
        seconds_remaining = self.T_in_seconds - seconds_elapsed
        time_remaining_years = max(0, seconds_remaining / self.seconds_in_year)

        # --- 2. Calculate Target Hedge Size ---
        S = self.data.Close[-1]
        _, delta = black_scholes_call(S, self.K, time_remaining_years, self.r, self.sigma)
        target_hedge_size = self.n_options * delta

        # --- 3. Rebalance Position ---
        current_hedge_size = self.position.size if self.position.is_long else 0
        trade_size = target_hedge_size - current_hedge_size

        # A small threshold to prevent churning from tiny delta changes
        if abs(trade_size) < 1: # Trade only if the change is at least 1 full unit
             return

        # Cast to int for unit-based sizing, as required by backtesting.py
        trade_units = int(trade_size)

        if trade_units > 0:
            self.buy(size=trade_units)
        elif trade_units < 0:
            # If we need to reduce our hedge, sell the difference
            self.sell(size=abs(trade_units))

        # --- 4. Close hedge at expiry ---
        if time_remaining_years == 0 and self.position:
            self.position.close()

if __name__ == '__main__':
    import os
    import json
    from backtesting import Backtest

    # --- Data Loading and Preparation ---
    # It is essential to use a long-enough dataset to cover the option's maturity
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure the data is available.")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Robustly clean column names (e.g., strip whitespace, title case)
    data.columns = [col.strip().title() for col in data.columns]
    # Remove any unnamed columns that might be created by pandas
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Ensure the data covers the entire option lifetime
    strategy_params = BlackScholesDeltaHedging
    required_days = strategy_params.T_days
    if len(data.index.normalize().unique()) < required_days:
        print(f"Warning: Data length ({len(data.index.normalize().unique())} days) is less than the option maturity ({required_days} days). Results may be partial.")
        # For this example, we'll proceed, but in a real case, you'd need more data.

    # Slice the data to match the option's lifetime for a clean backtest
    end_date = data.index[0] + pd.Timedelta(days=required_days)
    data = data[data.index <= end_date]

    # --- Backtest Execution ---
    # Use Backtest for unit-based sizing, which is required for this strategy.
    # Set a high cash value to support a large hedge on a high-priced asset like BTC.
    bt = Backtest(data, BlackScholesDeltaHedging, cash=100_000_000, commission=.002, finalize_trades=True)

    print("Running backtest...")
    stats = bt.run()
    print("Backtest complete.")

    # --- Results and Output ---
    print(stats)

    # Save results to a JSON file
    os.makedirs('results', exist_ok=True)
    strategy_name = 'black_scholes_european_option_delta_hedging'

    def sanitize_for_json(stats_obj):
        # Create a new dict to avoid modifying the original stats object
        sanitized = {}
        # Iterate over a copy of items to handle deletions safely
        for key, value in list(stats_obj.items()):
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                # Skip or convert these types if they are not needed
                continue
            if isinstance(value, (np.int64, np.integer)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                sanitized[key] = float(value)
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                # DataFrames and Series are not directly serializable
                continue
            elif pd.isna(value):
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

    clean_stats = sanitize_for_json(stats)

    result = {
        'strategy_name': strategy_name,
        'return': clean_stats.get('Return [%]'),
        'sharpe': clean_stats.get('Sharpe Ratio'),
        'max_drawdown': clean_stats.get('Max. Drawdown [%]'),
        'win_rate': clean_stats.get('Win Rate [%]'),
        'total_trades': clean_stats.get('# Trades')
    }

    # Use the requested filename 'temp_result.json'
    json_path = 'results/temp_result.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {json_path}")

    # Generate an HTML plot with a descriptive name
    plot_path = f'results/{strategy_name}.html'
    try:
        bt.plot(filename=plot_path)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
