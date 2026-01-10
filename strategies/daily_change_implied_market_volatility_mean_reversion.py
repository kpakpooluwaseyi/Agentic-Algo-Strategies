
from backtesting import Strategy
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import json
import os

# --- Helper Functions for Factor Model Simulation ---

def simulate_vix(btc_data):
    """
    Creates a synthetic VIX-like series from BTC daily price changes.
    """
    daily_returns = btc_data['Close'].resample('D').last().pct_change().abs()
    vix = daily_returns.rolling(window=21).std() * np.sqrt(365) # Annualized volatility
    vix = vix.ffill().resample('15T').ffill()
    return vix

def simulate_stock_universe(btc_data, vix, n_stocks=100):
    """
    Generates a universe of synthetic stocks with varying sensitivity to the VIX.
    """
    stocks = {}
    vix_impact_base = vix.pct_change().fillna(0)

    for i in range(n_stocks):
        # Assign a random beta to the VIX
        beta = np.random.uniform(-1.5, 1.5)

        # Create stock returns with a random component and a VIX-dependent component
        random_returns = pd.Series(np.random.normal(0, 0.02, len(btc_data)), index=btc_data.index)
        vix_impact = vix_impact_base * beta

        # Align series before operation
        stock_returns, vix_impact = random_returns.align(vix_impact, join='inner')

        combined_returns = stock_returns + vix_impact

        # Create the stock price series
        stocks[f'stock_{i}'] = (1 + combined_returns).cumprod() * btc_data['Close'][0]

    return pd.DataFrame(stocks)

def preprocess_data_factor_model(df, n_stocks=100, lookback=60):
    """
    Pre-processes the data to simulate the factor model and generate trading signals.
    """
    # 1. Simulate VIX and stock universe
    vix = simulate_vix(df)
    stocks = simulate_stock_universe(df, vix, n_stocks)

    # 2. Calculate factor loadings (betas)
    signals = []
    end_of_months = df.resample('M').last().index

    for month_end in end_of_months:
        start_date = month_end - pd.DateOffset(days=lookback)

        vix_returns = vix.loc[start_date:month_end].pct_change().dropna().values.reshape(-1, 1)

        betas = {}
        for stock_name in stocks.columns:
            stock_returns = stocks[stock_name].loc[start_date:month_end].pct_change().dropna().values

            # Ensure data is aligned
            min_len = min(len(vix_returns), len(stock_returns))
            vix_returns_aligned = vix_returns[-min_len:]
            stock_returns_aligned = stock_returns[-min_len:]

            if len(stock_returns_aligned) < 2:
                continue

            model = LinearRegression()
            model.fit(vix_returns_aligned, stock_returns_aligned)
            betas[stock_name] = model.coef_[0]

        if not betas:
            continue

        # 3. Rank stocks into quintiles
        beta_series = pd.Series(betas)
        quintiles = pd.qcut(beta_series, 5, labels=False, duplicates='drop')

        # 4. Generate signals for the main asset (BTC, proxied by stock_0)
        btc_quintile = quintiles.get('stock_0')
        signal = 0 # Neutral
        if btc_quintile == 0: # Lowest quintile
            signal = 1 # Buy
        elif btc_quintile == 4: # Highest quintile
            signal = -1 # Short

        signals.append({'datetime': month_end, 'signal': signal})

    signal_df = pd.DataFrame(signals).set_index('datetime')
    df = df.join(signal_df)
    df['signal'] = df['signal'].ffill().fillna(0)
    return df


# --- Strategy Class ---

class MoonDevStrategy(Strategy):
    pass

class DailyChangeImpliedMarketVolatilityMeanReversion(MoonDevStrategy):

    def init(self):
        # The signal is pre-calculated, so we just need to access it
        self.signal = self.I(lambda: self.data.df['signal'], name="Signal")

    def next(self):
        # Exit any open position at the end of the month
        if self.data.index[-1].month != self.data.index[-2].month:
            self.position.close()

        # Enter new position based on the monthly signal
        if not self.position:
            if self.signal[-1] == 1:
                self.buy()
            elif self.signal[-1] == -1:
                self.sell()

# --- Standalone Execution ---

if __name__ == '__main__':
    from backtesting import Backtest

    os.makedirs('results', exist_ok=True)

    # Load raw data
    try:
        df = pd.read_csv(
            'data/BTC-USD-15m.csv',
            header=None,
            skiprows=1,
            names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unnamed'],
            usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    except FileNotFoundError:
        print("Data file not found.")
        exit(1)

    # Pre-process data with the factor model simulation
    df_processed = preprocess_data_factor_model(df)

    bt = Backtest(df_processed, DailyChangeImpliedMarketVolatilityMeanReversion, cash=100_000, commission=.002)
    stats = bt.run()

    # --- JSON Serialization ---
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (pd.Timestamp, pd.Timedelta)): return str(obj)
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if hasattr(obj, 'to_dict'): return obj.to_dict()
            return super(CustomEncoder, self).default(obj)

    sanitized_stats = {k: v for k, v in stats.items() if not k.startswith('_')}

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4, cls=CustomEncoder)

    print(stats)

    try:
        bt.plot(filename='results/daily_change_implied_market_volatility_mean_reversion.html', open_browser=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")
