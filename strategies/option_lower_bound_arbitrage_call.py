import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import json

def generate_synthetic_data():
    """
    Generates a synthetic dataset to test the Option Lower Bound Arbitrage strategy.
    This function creates a proxy for the required data, including the underlying
    asset price (S), a synthetic call option price (C), and other parameters.
    Crucially, it programmatically introduces arbitrage opportunities where C drops
    below its theoretical lower bound.
    """
    n_points = 500
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))

    # 1. Generate Underlying Asset Price (S)
    s_initial = 100
    s_drift = 0.0001
    s_volatility = 0.005
    random_returns = np.random.normal(s_drift, s_volatility, n_points)
    close_np = s_initial * np.exp(np.cumsum(random_returns))

    data = pd.DataFrame(index=index)
    data['Close'] = close_np
    data['Open'] = data['Close'].shift(1).fillna(s_initial)
    data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, n_points))
    data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, n_points))
    data['Volume'] = np.random.randint(100, 1000, n_points)

    # 2. Define Option Parameters and add them to the DataFrame
    # In a real scenario, this data would come from an options data feed.
    data['Strike'] = 102  # K
    data['Risk_Free_Rate'] = 0.05 # r (annualized)

    # Time to maturity (T-t), decreasing over time, in years.
    total_duration_years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 3600)
    data['Time_To_Maturity'] = np.linspace(total_duration_years, 0.001, n_points)

    # 3. Generate Synthetic Call Price (C) and the Theoretical Lower Bound
    # The strategy needs to check if C < S - PV(K)
    pv_k = data['Strike'] * np.exp(-data['Risk_Free_Rate'] * data['Time_To_Maturity'])
    lower_bound = (data['Close'] - pv_k).clip(lower=0)

    # Store the lower bound in the DataFrame for the strategy to access.
    data['Lower_Bound'] = lower_bound

    # The call price should generally be above the lower bound due to time value, etc.
    # We add a small premium that decays as maturity approaches.
    premium = lower_bound * np.linspace(0.15, 0.01, n_points)
    data['Call_Price'] = lower_bound + premium + np.random.uniform(0.01, 0.05, n_points)

    # 4. Inject Arbitrage Opportunities
    # At specific points, force the Call_Price below the Lower_Bound to test the strategy.
    arbitrage_indices = [100, 250, 400]
    for idx in arbitrage_indices:
        # Force the underlying price to be high enough to create a valid lower bound
        data.at[data.index[idx], 'Close'] = data['Strike'].iloc[idx] + 5

        # Recalculate PV(K) and Lower Bound for this specific point
        pv_k_idx = data['Strike'].iloc[idx] * np.exp(-data['Risk_Free_Rate'].iloc[idx] * data['Time_To_Maturity'].iloc[idx])
        lower_bound_idx = data['Close'].iloc[idx] - pv_k_idx
        data.at[data.index[idx], 'Lower_Bound'] = lower_bound_idx

        # Now, inject the arbitrage opportunity
        if lower_bound_idx > 0.5: # This condition should now be met
            data.at[data.index[idx], 'Call_Price'] = lower_bound_idx - 0.2

    # The backtesting framework expects specific column names.
    # The 'Close' column will represent our underlying's spot price S(t).
    # The other required columns (Call_Price, Lower_Bound, etc.) are custom and will be
    # accessed via self.data.df in the strategy.
    return data


class OptionLowerBoundArbitrageCall(Strategy):
    """
    This strategy implements a proxy for a theoretical arbitrage opportunity.
    It identifies moments where a call option's price falls below its
    theoretical lower bound [max(0, S - PV(K))].

    Since the backtesting framework cannot model a multi-leg portfolio
    (long call, short stock, lending cash), this strategy uses a simplified
    proxy:
    - It buys the underlying asset when the arbitrage condition is met.
    - It closes the position on the next bar to simulate an immediate,
      risk-free profit capture.

    The performance metrics from this backtest (e.g., Return %) do not
    represent the actual arbitrage profit but confirm that the strategy's
    logic correctly identifies the synthetically created opportunities.
    """
    def init(self):
        # No indicators needed as the data is pre-calculated.
        # Caching data series in init can lead to issues during optimization,
        # so it's safer to access them directly in next().
        pass

    def next(self):
        # 1. Close any open position on the next bar.
        # This simulates the immediate capture of the arbitrage profit. The
        # actual profit is the initial cash inflow from setting up the
        # portfolio, which is not modeled here.
        if self.position:
            self.position.close()
            return

        # 2. Check for the arbitrage condition.
        # The core of the strategy: is the observed call price below its
        # theoretical minimum value?
        # We access the data directly from `self.data` for robustness.
        observed_call_price = self.data.Call_Price[-1]
        theoretical_lower_bound = self.data.Lower_Bound[-1]

        if observed_call_price < theoretical_lower_bound:
            # 3. If the condition is met, execute the proxy trade.
            # We buy the underlying asset as a signal that an arbitrage
            # opportunity was identified and acted upon.
            # No Stop-Loss or Take-Profit is set, as the position is
            # closed programmatically on the next bar.
            self.buy()


if __name__ == '__main__':
    # Generate the synthetic dataset with arbitrage opportunities
    data = generate_synthetic_data()

    # --- Backtest Configuration ---
    # Since this is a theoretical arbitrage model, we use a single run.
    # Optimization is not applicable.
    bt = Backtest(data, OptionLowerBoundArbitrageCall, cash=1_000_000, commission=.000)

    print("Running backtest...")
    stats = bt.run()
    print(stats)

    # --- Result Sanitization and Output ---
    def sanitize_stats(stats):
        """
        Cleans the backtesting stats object by removing non-serializable
        items and converting numpy types to native Python types.
        """
        if stats is None:
            return {}

        # Start with a copy of the stats Series
        result = {
            'strategy_name': 'option_lower_bound_arbitrage_call',
            'return_pct': stats.get('Return [%]', None),
            'sharpe_ratio': stats.get('Sharpe Ratio', None),
            'max_drawdown_pct': stats.get('Max. Drawdown [%]', None),
            'win_rate_pct': stats.get('Win Rate [%]', None),
            'num_trades': stats.get('# Trades', 0)
        }

        # Clean up values to ensure they are JSON serializable
        for key, value in result.items():
            if pd.isna(value):
                result[key] = None
            elif isinstance(value, (np.integer, np.int64)):
                result[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                result[key] = float(value)
        return result

    cleaned_results = sanitize_stats(stats)

    # Ensure the 'results' directory exists
    import os
    os.makedirs('results', exist_ok=True)

    # Save the sanitized results to a JSON file
    results_filepath = 'results/temp_result.json'
    with open(results_filepath, 'w') as f:
        json.dump(cleaned_results, f, indent=4)

    print(f"\nBacktest stats saved to {results_filepath}")

    # Generate and save the plot, wrapped in a try/except block for robustness
    plot_filepath = 'results/option_lower_bound_arbitrage_call.html'
    try:
        bt.plot(filename=plot_filepath)
        print(f"Plot saved to {plot_filepath}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
