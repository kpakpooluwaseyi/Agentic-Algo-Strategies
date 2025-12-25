
import numpy as np
import pandas as pd
from backtesting import Strategy
from scipy.stats import norm
from backtesting.lib import FractionalBacktest

def black_scholes_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def passthrough(series):
    return series

class CrrEuropeanOptionDeltaHedgingStrategy(Strategy):
    strike_price = 16600
    time_to_maturity_days = 30
    risk_free_rate = 0.02
    volatility_lookback = 10

    def init(self):
        self.S = self.I(passthrough, self.data.df['Close'].values, name="Close_Price")
        self.sigma = self.I(passthrough, self.data.df['AnnualizedVolatility'].values, name="Volatility")

        # Define the simulation start date to be after the volatility lookback period
        self.sim_start_date = self.data.index[0] + pd.Timedelta(days=self.volatility_lookback)
        self.maturity_date = self.sim_start_date + pd.Timedelta(days=self.time_to_maturity_days)
        self.last_rebalance_date = None

    def next(self):
        current_time = self.data.index[-1]
        current_date = current_time.date()

        # Don't start trading until the volatility warm-up is complete
        if current_time < self.sim_start_date:
            return

        if not (self.sim_start_date <= current_time <= self.maturity_date):
            if self.position:
                self.position.close()
            return

        if current_date == self.last_rebalance_date:
            return

        time_to_maturity_years = (self.maturity_date - current_time).total_seconds() / (365 * 24 * 3600)

        S_val = self.S[-1]
        sigma_val = self.sigma[-1]

        if S_val <= 0 or sigma_val <= 0 or time_to_maturity_years <= 0:
            if self.position: self.position.close()
            return

        target_delta_units = black_scholes_delta(S_val, self.strike_price, time_to_maturity_years, self.risk_free_rate, sigma_val)

        self.last_rebalance_date = current_date

        if target_delta_units <= 0:
            if self.position: self.position.close()
            return

        target_position_value = target_delta_units * S_val
        target_position_fraction = target_position_value / self.equity

        if self.position:
            self.position.close()

        if target_position_fraction > 1e-6:
            self.buy(size=target_position_fraction)

if __name__ == '__main__':
    import os
    import json

    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.strip().title() for c in data.columns]

    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    window_size = CrrEuropeanOptionDeltaHedgingStrategy.volatility_lookback * 24 * 4
    rolling_std = log_returns.rolling(window=window_size).std()
    annualized_volatility = rolling_std * np.sqrt(365 * 24 * 4)
    data['AnnualizedVolatility'] = annualized_volatility.fillna(0)

    bt = FractionalBacktest(data, CrrEuropeanOptionDeltaHedgingStrategy, cash=10_000_000, commission=.002)
    stats = bt.run()
    print(stats)

    os.makedirs('results', exist_ok=True)
    stats_dict = stats.to_dict()
    for key in ['_strategy', '_equity_curve', '_trades']:
        if key in stats_dict:
            del stats_dict[key]

    # Custom serializer to handle specific types
    def default_serializer(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.float64, np.float32)):
            return float(o)
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, pd.Timedelta):
            return str(o)
        if pd.isna(o):
            return None
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=2, default=default_serializer)

    bt.plot(filename='results/crr_european_option_delta_hedging.html')
