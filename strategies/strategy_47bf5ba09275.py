from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

def sanitize_stats(stats):
    """
    Sanitizes the backtesting stats object to be JSON serializable.
    Removes non-serializable objects like _strategy and _equity_curve.
    """
    if stats is None:
        return {}

    # Create a copy to avoid modifying the original object
    sanitized = stats.copy()

    # Remove non-serializable items
    if '_strategy' in sanitized:
        del sanitized['_strategy']
    if '_equity_curve' in sanitized:
        del sanitized['_equity_curve']
    if '_trades' in sanitized:
        del sanitized['_trades']

    # Convert pandas Timestamps/Timedeltas to strings
    for key, value in sanitized.items():
        if isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = value.item()
        elif pd.isna(value):
            sanitized[key] = None

    return sanitized


class ParabolicSarEmaTrendFollowing(Strategy):
    def init(self):
        pass

    def next(self):
        price = self.data.Close[-1]
        equity = self.equity

        # Long entry conditions
        is_uptrend = price > self.ema[-1]
        psar_flips_below = self.psar[-1] < self.data.Low[-1] and self.psar[-2] > self.data.High[-2]

        if is_uptrend and psar_flips_below and not self.position:
            sl = self.psar[-1]
            if sl >= price: return

            # Risk management
            max_risk_sl = price * (1 - self.max_risk_pct)
            sl = max(sl, max_risk_sl) # Use the tighter of the two stop losses

            risk_per_unit = price - sl
            if risk_per_unit <= 0: return

            size = (equity * self.max_risk_pct) / risk_per_unit
            tp = price + (price - sl) * self.rr_ratio
            self.buy(size=size, sl=sl, tp=tp)

        # Short entry conditions
        is_downtrend = price < self.ema[-1]
        psar_flips_above = self.psar[-1] > self.data.High[-1] and self.psar[-2] < self.data.Low[-2]

        if is_downtrend and psar_flips_above and not self.position:
            sl = self.psar[-1]
            if sl <= price: return

            # Risk management
            max_risk_sl = price * (1 + self.max_risk_pct)
            sl = min(sl, max_risk_sl) # Use the tighter of the two stop losses

            risk_per_unit = sl - price
            if risk_per_unit <= 0: return

            size = (equity * self.max_risk_pct) / risk_per_unit
            tp = price - (sl - price) * self.rr_ratio
            self.sell(size=size, sl=sl, tp=tp)

if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
        # Sanitize column names
        df.columns = [c.strip().title() for c in df.columns]
    except FileNotFoundError:
        print("No data file found. Using sample data generation...")
        # Generate simple sample data
        dates = pd.date_range('2023-01-01', periods=5000, freq='15min')
        price = 20000 + pd.Series(np.random.randn(5000).cumsum() * 10)
        df = pd.DataFrame({
            'Open': price,
            'High': price + pd.Series(np.random.rand(5000) * 20),
            'Low': price - pd.Series(np.random.rand(5000) * 20),
            'Close': price + pd.Series(np.random.randn(5000) * 5),
            'Volume': pd.Series(np.random.rand(5000) * 1000)
        }, index=dates)


    # Run backtest
    bt = Backtest(df, ParabolicSarEmaTrendFollowing, cash=100_000, commission=.002)
    stats = bt.run()

    # Sanitize and save results
    sanitized_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print("=== Backtest Results ===")
    print(stats)

    # Save plot
    bt.plot(filename='results/strategy_47bf5ba09275.html', open_browser=False)
    print("\nPlot saved to results/strategy_47bf5ba09275.html")
