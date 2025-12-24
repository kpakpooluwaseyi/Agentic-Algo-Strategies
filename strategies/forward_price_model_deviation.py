from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd
import json
import os

def preprocess_data_with_forecasts(df, r, delta, maturity_days):
    """
    Pre-calculates the theoretical forward prices for each bar.
    This is more efficient than calculating it iteratively inside the strategy.
    """
    # Calculate the number of bars in one day from the data frequency
    freq = pd.infer_freq(df.index[:10])
    time_delta = pd.to_timedelta(freq) if freq else pd.Timedelta(minutes=15) # Fallback
    bars_per_day = int(pd.Timedelta(days=1) / time_delta)
    maturity_bars = maturity_days * bars_per_day

    # T_minus_t: Time to maturity in years
    T_minus_t = maturity_days / 365.25

    # Calculate theoretical forward price for each point in time
    # This is the price that is *forecasted* at time `t` for time `t + maturity_bars`
    df['Forecast'] = df['Close'] * np.exp((r - delta) * T_minus_t)

    # Shift the forecast back by the maturity period so that at any given index,
    # df['Forecast'] contains the price that was forecasted for it in the past.
    df['Past_Forecast'] = df['Forecast'].shift(maturity_bars)

    return df.dropna()


class ForwardPriceModelDeviationStrategy(Strategy):
    """
    This strategy adapts the Forward-Spot Arbitrage concept by using the
    theoretical forward price formula as a predictive model for future spot prices.

    1.  At each bar `t`, it calculates a theoretical forward price for a future bar
        `t + maturity_period`. This acts as a forecast.
    2.  When the future bar `t + maturity_period` is reached, it compares the
        *actual* spot price against the price that was forecasted for it in the past.
    3.  If the actual spot price has deviated significantly from the past forecast,
        a trade is initiated with the assumption that the price will mean-revert
        towards the model's prediction.
    """
    # --- Optimizable Parameters ---
    r = 0.05  # Annualized risk-free rate
    delta = 0.02  # Annualized dividend/yield
    maturity_days = 30  # Maturity of the synthetic forward contract
    deviation_threshold_pct = 5.0 # Required deviation to trigger a trade
    hold_bars = 12 # How many bars to hold the position after entry

    def init(self):
        self.past_forecast = self.I(lambda x: x, self.data.df['Past_Forecast'].values)
        self.trade_entry_bar = None

    def next(self):
        current_bar = len(self.data.Close) - 1

        # --- Position Management: Time-based Exit ---
        if self.position:
            if self.trade_entry_bar is not None and (current_bar - self.trade_entry_bar) >= self.hold_bars:
                self.position.close()
                self.trade_entry_bar = None
            return

        # --- Entry Logic ---
        if self.past_forecast[-1] is None or np.isnan(self.past_forecast[-1]):
            return

        actual_price = self.data.Close[-1]
        forecasted_price = self.past_forecast[-1]

        deviation = (actual_price - forecasted_price) / forecasted_price * 100

        # If actual price is significantly HIGHER than forecast, expect it to revert down (SELL)
        if deviation > self.deviation_threshold_pct:
            self.sell()
            self.trade_entry_bar = current_bar

        # If actual price is significantly LOWER than forecast, expect it to revert up (BUY)
        elif deviation < -self.deviation_threshold_pct:
            self.buy()
            self.trade_entry_bar = current_bar

def sanitize_stats(stats):
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Series, pd.DataFrame, Strategy)):
            continue
        if pd.isna(value):
            sanitized[key] = None
        elif isinstance(value, (int, float)):
            sanitized[key] = value
        elif isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        else:
            sanitized[key] = value
    sanitized.pop('_strategy', None)
    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        print(f"Loading data from: {data_path}")
        data_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data_df.columns = [c.strip().capitalize() for c in data_df.columns]
        data_df = data_df.loc[:, ~data_df.columns.str.contains('^Unnamed')]

        # Preprocess data to include forecasts
        data_with_forecasts = preprocess_data_with_forecasts(
            data_df,
            r=ForwardPriceModelDeviationStrategy.r,
            delta=ForwardPriceModelDeviationStrategy.delta,
            maturity_days=ForwardPriceModelDeviationStrategy.maturity_days
        )

        print("Initializing Backtest...")
        bt = Backtest(data_with_forecasts, ForwardPriceModelDeviationStrategy, cash=100_000, commission=.002)

        print("Running backtest with default parameters...")
        stats = bt.run()
        print(stats)

        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        clean_stats = sanitize_stats(stats)

        result_data = {
            'strategy_name': 'forward_price_model_deviation',
            'parameters': {
                'r': ForwardPriceModelDeviationStrategy.r,
                'delta': ForwardPriceModelDeviationStrategy.delta,
                'maturity_days': ForwardPriceModelDeviationStrategy.maturity_days,
                'deviation_threshold_pct': ForwardPriceModelDeviationStrategy.deviation_threshold_pct,
                'hold_bars': ForwardPriceModelDeviationStrategy.hold_bars
            },
            'stats': clean_stats
        }

        results_filepath = os.path.join(results_dir, 'temp_result.json')
        print(f"Saving results to {results_filepath}")
        with open(results_filepath, 'w') as f:
            json.dump(result_data, f, indent=2)

        plot_filepath = os.path.join(results_dir, 'forward_price_model_deviation.html')
        print(f"Generating plot and saving to {plot_filepath}")
        try:
            bt.plot(filename=plot_filepath, open_browser=False)
        except Exception as e:
            print(f"Could not generate plot. Error: {e}")
