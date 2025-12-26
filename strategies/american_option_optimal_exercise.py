
import numpy as np
import pandas as pd
import os

class AmericanOptionSimulator:
    """
    Simulates the optimal exercise of an American option using a binomial pricing model.
    This is a financial model, not a trading strategy to be run with backtesting.py.
    """

    def __init__(self, S, K, T, r, sigma, N, option_type='call'):
        """
        Initializes the American Option Simulator.

        Args:
            S (float): Current price of the underlying asset.
            K (float): Strike price of the option.
            T (float): Time to maturity (in years).
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            N (int): Number of steps in the binomial tree.
            option_type (str): 'call' or 'put'.
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.option_type = option_type.lower()

        # Calculate binomial tree parameters
        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)

        if not (self.d < np.exp(self.r * self.dt) < self.u):
            raise ValueError("Arbitrage opportunity detected in parameters. "
                             "Ensure d < e^(r*dt) < u.")

        self.stock_price_tree = self._build_stock_price_tree()
        self.option_value_tree = self._calculate_option_value_tree()

    def _build_stock_price_tree(self):
        """Builds the binomial tree for the underlying stock price."""
        tree = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.N + 1):
            for j in range(i + 1):
                tree[j, i] = self.S * (self.u ** (i - j)) * (self.d ** j)
        return tree

    def _calculate_option_value_tree(self):
        """Calculates the option value at each node using backward induction."""
        tree = np.zeros((self.N + 1, self.N + 1))

        # Payoff at maturity (time N)
        if self.option_type == 'call':
            tree[:, self.N] = np.maximum(0, self.stock_price_tree[:, self.N] - self.K)
        else:  # put
            tree[:, self.N] = np.maximum(0, self.K - self.stock_price_tree[:, self.N])

        # Backward induction
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                future_value = np.exp(-self.r * self.dt) * (
                    self.p * tree[j, i + 1] + (1 - self.p) * tree[j + 1, i + 1]
                )

                if self.option_type == 'call':
                    exercise_value = np.maximum(0, self.stock_price_tree[j, i] - self.K)
                else:  # put
                    exercise_value = np.maximum(0, self.K - self.stock_price_tree[j, i])

                tree[j, i] = np.maximum(exercise_value, future_value)

        return tree

    def get_option_price(self):
        """Returns the calculated price of the American option at t=0."""
        return self.option_value_tree[0, 0]

    def get_exercise_decision_boundary(self):
        """
        Returns a DataFrame showing the stock price at which it is optimal to exercise
        at each time step.
        """
        exercise_boundary = []
        for i in range(self.N, -1, -1):
            boundary_price = None
            for j in range(i + 1):
                future_value = np.exp(-self.r * self.dt) * (
                    self.p * self.option_value_tree[j, i + 1] +
                    (1 - self.p) * self.option_value_tree[j + 1, i + 1]
                ) if i < self.N else 0

                if self.option_type == 'call':
                    exercise_value = np.maximum(0, self.stock_price_tree[j, i] - self.K)
                    if exercise_value > future_value and exercise_value > 1e-6:
                        boundary_price = self.stock_price_tree[j, i]
                        # For calls, we care about the highest price to exercise
                        break
                else:  # put
                    exercise_value = np.maximum(0, self.K - self.stock_price_tree[j, i])
                    if exercise_value > future_value and exercise_value > 1e-6:
                        # For puts, we care about the lowest price to exercise
                        boundary_price = self.stock_price_tree[j, i]

            exercise_boundary.append({'time_step': i, 'exercise_price_boundary': boundary_price})

        df = pd.DataFrame(exercise_boundary).sort_values('time_step').set_index('time_step')
        df['time_in_years'] = df.index * self.dt
        return df


if __name__ == '__main__':
    # --- 1. Load Data and Calculate Parameters ---
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure the data is available.")

    # Load data
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    df.columns = [x.strip().capitalize() for x in df.columns]

    # Get the latest price as the starting stock price (S)
    S0 = df['Close'].iloc[-1]

    # Calculate historical volatility (sigma) from 15-minute data
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    # Number of 15-min periods in a year = 365 days * 24 hours * 4 quarters
    annualization_factor = np.sqrt(365 * 24 * 4)
    sigma = log_returns.std() * annualization_factor

    # --- 2. Define Option Parameters ---
    K = S0 * 1.05  # Strike price 5% out-of-the-money
    T = 0.25       # Time to maturity: 3 months (in years)
    r = 0.05       # Risk-free rate: 5%
    N = 100        # Number of steps in the binomial tree

    print("--- American Option Optimal Exercise Simulation ---")
    print(f"Underlying Asset: BTC-USD")
    print(f"Current Price (S): ${S0:,.2f}")
    print(f"Calculated Annual Volatility (sigma): {sigma:.2%}")
    print(f"Strike Price (K): ${K:,.2f}")
    print(f"Time to Maturity (T): {T} years")
    print(f"Risk-Free Rate (r): {r:.2%}")
    print(f"Binomial Steps (N): {N}")
    print("-" * 50)

    try:
        # --- 3. Simulate American Call Option ---
        print("\n--- Simulating American CALL Option ---")
        call_sim = AmericanOptionSimulator(S=S0, K=K, T=T, r=r, sigma=sigma, N=N, option_type='call')
        call_price = call_sim.get_option_price()
        call_boundary = call_sim.get_exercise_decision_boundary()

        print(f"Calculated Call Option Price: ${call_price:,.2f}")
        print("Optimal Exercise Boundary (Stock Price):")
        # Display the first few and last few steps for brevity
        print(call_boundary.head(5).to_string())
        print("...")
        print(call_boundary.tail(5).to_string())

        # --- 4. Simulate American Put Option ---
        print("\n--- Simulating American PUT Option ---")
        put_sim = AmericanOptionSimulator(S=S0, K=K, T=T, r=r, sigma=sigma, N=N, option_type='put')
        put_price = put_sim.get_option_price()
        put_boundary = put_sim.get_exercise_decision_boundary()

        print(f"Calculated Put Option Price: ${put_price:,.2f}")
        print("Optimal Exercise Boundary (Stock Price):")
        print(put_boundary.head(5).to_string())
        print("...")
        print(put_boundary.tail(5).to_string())

    except ValueError as e:
        print(f"\nError during simulation: {e}")
