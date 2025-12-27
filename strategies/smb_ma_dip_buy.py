from backtesting import Strategy
import pandas_ta as ta
import pandas as pd

def sma(arr: pd.Series, n: int) -> pd.Series:
    """Custom SMA indicator using pandas-ta"""
    return ta.sma(arr, length=n)

class SmbMaDipBuyStrategy(Strategy):
    """
    Strategy that buys dips against rising 5-day and 10-day Simple Moving Averages (SMAs).
    """
    # Optimizable parameters
    fast_sma_period = 5
    slow_sma_period = 10
    rr_ratio = 2.0

    def init(self):
        """
        Initialize indicators and strategy variables.
        """
        # Initialize the SMAs
        self.fast_sma = self.I(sma, pd.Series(self.data.Close), self.fast_sma_period)
        self.slow_sma = self.I(sma, pd.Series(self.data.Close), self.slow_sma_period)

    def next(self):
        """
        Defines the trading logic for each bar.
        """
        price = self.data.Close[-1]

        # --- Trend Condition ---
        # Both SMAs must be rising
        fast_sma_rising = self.fast_sma[-1] > self.fast_sma[-2]
        slow_sma_rising = self.slow_sma[-1] > self.slow_sma[-2]

        # Price should be above the slow SMA for a clear uptrend
        price_above_slow_sma = price > self.slow_sma[-1]

        # --- Entry Logic ---
        if not self.position and fast_sma_rising and slow_sma_rising and price_above_slow_sma:
            # Dip Condition: Price pulls back to touch the fast SMA
            if self.data.Low[-1] <= self.fast_sma[-1] and price > self.fast_sma[-1]:

                # --- Risk Management ---
                entry_price = price
                stop_loss = self.slow_sma[-1]
                take_profit = entry_price + (entry_price - stop_loss) * self.rr_ratio

                # Ensure stop loss is valid before placing a trade
                if entry_price > stop_loss:
                    self.buy(size=1, sl=stop_loss, tp=take_profit)

if __name__ == '__main__':
    import os
    import json
    from backtesting import Backtest

    # --- Data Loading ---
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.strip().capitalize() for c in data.columns]
    # Drop unnamed columns that can be created by a trailing comma in the CSV header
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # --- Backtest Execution ---
    bt = Backtest(data, SmbMaDipBuyStrategy, cash=100_000, commission=.002)

    print("Running single backtest with default parameters...")
    stats = bt.run()

    # --- Results ---
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON output
    result = {
        'strategy_name': 'smb_ma_dip_buy',
        'return': float(stats.get('Return [%]', 0)),
        'sharpe': float(stats.get('Sharpe Ratio', 0)),
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)),
        'win_rate': float(stats.get('Win Rate [%]', 0)),
        'total_trades': int(stats.get('# Trades', 0))
    }

    # Save stats to JSON
    result_filename = 'results/temp_result.json'
    with open(result_filename, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Backtest results saved to {result_filename}")

    # Generate and save plot
    try:
        plot_filename = 'results/smb_ma_dip_buy.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
