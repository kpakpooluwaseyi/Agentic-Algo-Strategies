import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import json
import os

def preprocess_data_for_vix_proxy(df, futures_period=10):
    """
    Prepares the single-asset (BTC) data to be used as a proxy for the multi-asset
    VIX/ES strategy. This is a literal interpretation of a contradictory request.
    """
    df['VIX_Spot'] = df['Close']
    df['VIX_Futures'] = df['Close'].rolling(window=futures_period).mean()

    # Create a synthetic, negatively correlated hedging instrument
    df['ES_Futures'] = (df['Close'].max() - df['Close']) + df['Close'].mean()

    # Generate synthetic settlement days (daily timeframe logic on 15m data)
    days_in_cycle = 21  # Approx. business days in a month
    total_bars = len(df)
    # This creates a countdown from 21 down to 1, repeating.
    # We use a simple modulo approach for this synthetic daily countdown.
    bars_per_day = 96 # 24 hours * 4 15-min bars
    countdown = np.array([days_in_cycle - (i // bars_per_day) % days_in_cycle for i in range(total_bars)])
    df['Days_to_Settlement'] = countdown

    # Drop only the rows where the rolling mean created NaNs
    df.dropna(subset=['VIX_Futures'], inplace=True)
    return df

def simulate_hedged_portfolio_from_proxy(data, daily_roll_threshold=0.1, min_days_to_settlement=10):
    """
    Performs a vectorized simulation of the hedged VIX carry strategy using the proxy data.
    Returns a Series of daily portfolio returns.
    """
    # Use .loc to avoid SettingWithCopyWarning
    data = data.copy()
    data['Daily_Roll'] = (data['VIX_Futures'] - data['VIX_Spot']) / data['Days_to_Settlement']

    # Position state: -1 for short VIX/long ES, 1 for long VIX/short ES
    positions = pd.Series(0.0, index=data.index)

    contango_mask = (data['Daily_Roll'] > daily_roll_threshold) & (data['Days_to_Settlement'] >= min_days_to_settlement)
    backwardation_mask = (data['Daily_Roll'] < -daily_roll_threshold) & (data['Days_to_Settlement'] >= min_days_to_settlement)

    positions.loc[contango_mask] = -1
    positions.loc[backwardation_mask] = 1

    positions = positions.replace(0, method='ffill').fillna(0)

    vix_returns = data['VIX_Futures'].pct_change()
    es_returns = data['ES_Futures'].pct_change()

    # Long VIX (+1) is paired with Short ES (-1)
    # Short VIX (-1) is paired with Long ES (+1)
    portfolio_returns = (positions.shift(1) * vix_returns) - (positions.shift(1) * es_returns)

    return portfolio_returns.fillna(0)

class HedgedVixFuturesCarryStrategy(Strategy):
    """
    A simple buy-and-hold strategy used for analyzing a pre-computed equity curve.

    The actual complex logic (VIX term structure, hedging) is performed in the
    data pre-processing and simulation steps, as it involves multiple instruments
    that cannot be handled by a single `backtesting.py` Strategy instance. This
    class is a vehicle to use the framework's plotting and stats reporting tools.
    """
    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy()

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    strategy_name = 'hedged_vix_futures_carry_term_structure'
    initial_cash = 100_000

    print("1. Loading and pre-processing BTC data as a proxy for VIX/ES...")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.strip().title() for c in data.columns]
    proxy_data = preprocess_data_for_vix_proxy(data)

    print("2. Simulating the hedged portfolio to generate returns...")
    portfolio_returns = simulate_hedged_portfolio_from_proxy(proxy_data)

    print("3. Creating cumulative equity curve for backtest...")
    equity_curve = (1 + portfolio_returns).cumprod() * initial_cash

    ohlc_data = pd.DataFrame(index=equity_curve.index)
    ohlc_data['Open'] = equity_curve.shift(1).fillna(initial_cash)
    ohlc_data['High'] = equity_curve.shift(1).fillna(initial_cash) # To show movement
    ohlc_data['Low'] = equity_curve.shift(1).fillna(initial_cash)
    ohlc_data['Close'] = equity_curve
    ohlc_data['Volume'] = 0

    print("4. Running backtest on the final equity curve...")
    bt = Backtest(ohlc_data, HedgedVixFuturesCarryStrategy, cash=initial_cash, commission=0.0)
    stats = bt.run()
    print(stats)

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(key, str) and not key.startswith('_'):
                if isinstance(value, (np.integer, int)):
                    sanitized[key] = int(value)
                elif isinstance(value, (np.floating, float)):
                    sanitized[key] = float(value) if np.isfinite(value) else None
                else:
                    sanitized[key] = str(value)
        return sanitized

    final_stats = sanitize_stats(stats)
    results_filepath = os.path.join(results_dir, 'temp_result.json')
    with open(results_filepath, 'w') as f:
        json.dump(final_stats, f, indent=2)
    print(f"Backtest statistics saved to {results_filepath}")

    plot_filepath = os.path.join(results_dir, f"{strategy_name}.html")
    try:
        bt.plot(filename=plot_filepath, open_browser=False)
        print(f"Backtest plot saved to {plot_filepath}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
