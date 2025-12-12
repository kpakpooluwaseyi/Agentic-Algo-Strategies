import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

# ===================================================================================
# Synthetic Data Generation
# ===================================================================================
def generate_synthetic_data():
    """Generates synthetic data with textbook long and short setups."""
    # Short setup: Downtrend -> Anchor Low -> Pullback -> Entry
    short_setup_data = [
        {'Open': 105, 'High': 105.5, 'Low': 104, 'Close': 104.2},
        {'Open': 104.2, 'High': 104.5, 'Low': 103, 'Close': 103.1},
        {'Open': 103.1, 'High': 103.3, 'Low': 102, 'Close': 102.2},
        {'Open': 102.2, 'High': 102.5, 'Low': 101, 'Close': 101.5}, # C_Anchor
        {'Open': 101.5, 'High': 102.8, 'Low': 101.4, 'Close': 102.0},
        {'Open': 102.0, 'High': 103.5, 'Low': 101.9, 'Close': 103.2}, # Pullback High
        {'Open': 103.2, 'High': 103.4, 'Low': 102.5, 'Close': 102.8},
        {'Open': 102.8, 'High': 103.0, 'Low': 101.8, 'Close': 102.2}, # C_Entry
        {'Open': 102.3, 'High': 102.4, 'Low': 101.0, 'Close': 101.2}, # TP hit
    ]
    # Long setup: Uptrend -> Anchor High -> Pullback -> Entry
    long_setup_data = [
        {'Open': 110, 'High': 111, 'Low': 109.5, 'Close': 110.8},
        {'Open': 110.8, 'High': 112, 'Low': 110.5, 'Close': 111.9},
        {'Open': 112.8, 'High': 114, 'Low': 112.7, 'Close': 113.5}, # C_Anchor
        {'Open': 113.5, 'High': 113.6, 'Low': 112.5, 'Close': 112.8},
        {'Open': 112.8, 'High': 113.0, 'Low': 111.5, 'Close': 111.8}, # Pullback Low
        {'Open': 111.8, 'High': 112.4, 'Low': 111.6, 'Close': 112.2},
        {'Open': 112.2, 'High': 113.2, 'Low': 112.0, 'Close': 112.9}, # C_Entry
        {'Open': 112.8, 'High': 114.1, 'Low': 112.8, 'Close': 113.8}, # TP hit
    ]
    neutral_data = [{'Open': 100, 'High': 100.5, 'Low': 99.5, 'Close': 100}] * 10
    df = pd.DataFrame(neutral_data + short_setup_data + neutral_data + long_setup_data + neutral_data)
    df['Volume'] = 100
    start_time = pd.Timestamp('2023-01-02 07:00:00')
    df.index = pd.to_datetime([start_time + pd.Timedelta(minutes=i*5) for i in range(len(df))])
    return df

# ===================================================================================
# JSON Serialization Helper
# ===================================================================================
def sanitize_stats(stats):
    sanitized = {k: v for k, v in stats.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
    for key, value in sanitized.items():
        if isinstance(value, (np.integer, np.int64)): sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)): sanitized[key] = float(value)
    for k in ['Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]', 'Win Rate [%]', '# Trades']:
        if k not in sanitized: sanitized[key] = 0.0 if k != '# Trades' else 0
    return sanitized

# ===================================================================================
# Strategy Implementation
# ===================================================================================
class RangeBarPullbackContinuationStrategy(Strategy):
    min_rr = 1.0      # Optimizable R:R ratio
    risk_pct = 0.01   # Percentage of equity to risk
    pip_size = 0.0001 # Pip size for SL placement

    def init(self):
        self.trend = 0
        self.anchor_high, self.anchor_low, self.anchor_open = None, None, None
        self.pullback_high, self.pullback_low = None, None
        self.in_pullback = False
        self.trade_pending = False
        self.is_uk_session = self.I(lambda x: x, self.data.df['is_uk_session'], name="is_uk_session")

    def next(self):
        if self.position:
            self.trade_pending = False # Trade is active, no longer pending
            return

        # If a trade was placed but didn't open (e.g., due to slippage), reset pending state
        if self.trade_pending and not self.position:
            self.trade_pending = False

        if self.trade_pending:
            return

        close, high, low = self.data.Close[-1], self.data.High[-1], self.data.Low[-1]

        if not self.is_uk_session[-1]:
            self.trend = 0
            return

        # ==================
        # LONG TRADE LOGIC
        # ==================
        if self.trend != 1 and close > self.data.Close[-2]:
            self.trend, self.anchor_high, self.anchor_open, self.in_pullback = 1, close, self.data.Open[-1], False
        elif self.trend == 1:
            if close > self.anchor_high:
                self.anchor_high, self.anchor_open, self.in_pullback = close, self.data.Open[-1], False
            elif close < self.data.Close[-2]:
                if not self.in_pullback: self.in_pullback, self.pullback_low = True, low
                else: self.pullback_low = min(self.pullback_low, low)
            elif self.in_pullback and close > self.data.Close[-2]:
                entry, sl, tp = close, self.pullback_low - self.pip_size, self.anchor_open
                if entry > sl and (tp - entry) / (entry - sl) >= self.min_rr:
                    risk_units = entry - sl
                    calc_size = (self.equity * self.risk_pct) / risk_units
                    max_size = self.equity / entry
                    size = int(min(calc_size, max_size))
                    if size >= 1: self.buy(sl=sl, tp=tp, size=size); self.trade_pending = True

        # ==================
        # SHORT TRADE LOGIC
        # ==================
        elif self.trend != -1 and close < self.data.Close[-2]:
            self.trend, self.anchor_low, self.anchor_open, self.in_pullback = -1, close, self.data.Open[-1], False
        elif self.trend == -1:
            if close < self.anchor_low:
                self.anchor_low, self.anchor_open, self.in_pullback = close, self.data.Open[-1], False
            elif close > self.data.Close[-2]:
                if not self.in_pullback: self.in_pullback, self.pullback_high = True, high
                else: self.pullback_high = max(self.pullback_high, high)
            elif self.in_pullback and close < self.data.Close[-2]:
                entry, sl, tp = close, self.pullback_high + self.pip_size, self.anchor_open
                if sl > entry and (entry - tp) / (sl - entry) >= self.min_rr:
                    risk_units = sl - entry
                    calc_size = (self.equity * self.risk_pct) / risk_units
                    max_size = self.equity / entry
                    size = int(min(calc_size, max_size))
                    if size >= 1: self.sell(sl=sl, tp=tp, size=size); self.trade_pending = True

# ===================================================================================
# Backtest Execution Block
# ===================================================================================
if __name__ == '__main__':
    data = generate_synthetic_data()
    uk_start, uk_end = pd.to_datetime("07:30").time(), pd.to_datetime("08:30").time()
    data['is_uk_session'] = (data.index.time >= uk_start) & (data.index.time <= uk_end)

    bt = Backtest(data, RangeBarPullbackContinuationStrategy, cash=100_000, commission=.002)
    stats = bt.optimize(min_rr=np.arange(0.5, 1.5, 0.1).tolist(), maximize='Sharpe Ratio')

    print("Best Run Stats:")
    print(stats)
    print("\nTrades from the best run:")
    print(stats._trades)

    sanitized = sanitize_stats(stats)
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'range_bar_pullback_continuation',
            'return': sanitized.get('Return [%]'),
            'sharpe': sanitized.get('Sharpe Ratio'),
            'max_drawdown': sanitized.get('Max. Drawdown [%]'),
            'win_rate': sanitized.get('Win Rate [%]'),
            'total_trades': sanitized.get('# Trades')
        }, f, indent=2)

    try:
        bt.plot(filename='results/range_bar_pullback_continuation.html', open_browser=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")
