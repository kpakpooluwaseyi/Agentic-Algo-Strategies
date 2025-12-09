from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

class AsiaLiquidityGrabReversalStrategy(Strategy):
    """
    Strategy based on the Asia session liquidity grab with multiple take-profit levels
    and break-even stop-loss management.
    """

    # --- Optimizable Strategy Parameters ---
    asia_start_hour = 20
    asia_end_hour = 9
    volatility_threshold = 2.0
    trade_size = 0.99

    # --- Hardcoded Parameters ---
    asia_end_minute = 30

    # --- Indicators ---
    hoa = None
    loa = None
    volatility_ok = None
    daily_50_pct = None

    def init(self):
        df = self.data.df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        try:
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            else: df.index = df.index.tz_convert('UTC')
        except TypeError as e:
            print(f"Timestamp timezone handling notice: {e}")

        df['HOA'], df['LOA'], df['volatility_ok'], df['daily_50_pct'] = [np.nan, np.nan, False, np.nan]

        unique_days = df.index.normalize().unique()

        for i, day in enumerate(unique_days):
            if i == 0: continue

            session_end = day.replace(hour=self.asia_end_hour, minute=self.asia_end_minute)
            session_start = (day - pd.Timedelta(days=1)).replace(hour=self.asia_start_hour, minute=0)
            session_mask = (df.index >= session_start) & (df.index <= session_end)

            if not df[session_mask].empty:
                hoa, loa = df[session_mask]['High'].max(), df[session_mask]['Low'].min()
                volatility = ((hoa - loa) / loa) * 100
                is_ok = volatility < self.volatility_threshold

                trade_window = (df.index.normalize() == day) & (df.index > session_end)
                df.loc[trade_window, ['HOA', 'LOA', 'volatility_ok']] = [hoa, loa, is_ok]

            prev_day_mask = df.index.normalize() == unique_days[i-1]
            if prev_day_mask.any():
                prev_high, prev_low = df[prev_day_mask]['High'].max(), df[prev_day_mask]['Low'].min()
                df.loc[df.index.normalize() == day, 'daily_50_pct'] = prev_low + (prev_high - prev_low) / 2

        self.hoa = self.I(lambda: df['HOA'])
        self.loa = self.I(lambda: df['LOA'])
        self.volatility_ok = self.I(lambda: df['volatility_ok'])
        self.daily_50_pct = self.I(lambda: df['daily_50_pct'])

    def is_bullish_engulfing(self):
        if len(self.data.Close) < 2: return False
        return (self.data.Close[-2] < self.data.Open[-2] and self.data.Close[-1] > self.data.Open[-1] and
                self.data.Close[-1] > self.data.Open[-2] and self.data.Open[-1] < self.data.Close[-2])

    def is_bearish_engulfing(self):
        if len(self.data.Close) < 2: return False
        return (self.data.Close[-2] > self.data.Open[-2] and self.data.Close[-1] < self.data.Open[-1] and
                self.data.Open[-1] > self.data.Close[-2] and self.data.Close[-1] < self.data.Open[-2])

    def next(self):
        # --- Trade Management: Move SL to BE ---
        # If there's one trade left, and its T1 was hit (implied by one trade being closed)
        if len(self.trades) == 1:
            trade = self.trades[0]
            # If current price is roughly halfway to the original T1, move SL to BE.
            # This is a proxy since we can't know which TP was hit.
            # A more robust solution might need custom trade management logic.
            if trade.is_long and self.data.Close[-1] > (trade.entry_price + (self.loa[-1] - trade.entry_price) / 2):
                trade.sl = trade.entry_price
            elif trade.is_short and self.data.Close[-1] < (trade.entry_price - (trade.entry_price - self.hoa[-1]) / 2):
                trade.sl = trade.entry_price

        # --- Entry Logic ---
        if not self.volatility_ok[-1] or len(self.trades) > 0:
            return

        price = self.data

        if (price.High[-2] > self.hoa[-1] and self.is_bearish_engulfing() and price.Close[-1] < self.hoa[-1]):
            sl, tp1, tp2 = price.High[-1], self.loa[-1], self.daily_50_pct[-1]
            if tp1 < price.Close[-1]: self.sell(size=self.trade_size/2, sl=sl, tp=tp1)
            if not np.isnan(tp2) and tp2 < price.Close[-1]: self.sell(size=self.trade_size/2, sl=sl, tp=tp2)

        if (price.Low[-2] < self.loa[-1] and self.is_bullish_engulfing() and price.Close[-1] > self.loa[-1]):
            sl, tp1, tp2 = price.Low[-1], self.hoa[-1], self.daily_50_pct[-1]
            if tp1 > price.Close[-1]: self.buy(size=self.trade_size/2, sl=sl, tp=tp1)
            if not np.isnan(tp2) and tp2 > price.Close[-1]: self.buy(size=self.trade_size/2, sl=sl, tp=tp2)

if __name__ == '__main__':
    from backtesting.test import GOOG
    data = GOOG.copy()[-730:]
    data = data.resample('15min').ffill()

    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=10000, commission=.002)

    print("Starting optimization...")
    stats = bt.optimize(
        asia_start_hour=[20, 21],
        asia_end_hour=[8, 9],
        volatility_threshold=[2.0, 2.5],
        maximize='Sharpe Ratio'
    )
    print("Optimization complete.")
    print(stats)

    import os
    os.makedirs('results', exist_ok=True)

    if stats['# Trades'] > 0:
        stats_dict = {'strategy_name': 'asia_liquidity_grab_reversal (Optimized)',
                      'return': float(stats['Return [%]']), 'sharpe': float(stats['Sharpe Ratio']),
                      'max_drawdown': float(stats['Max. Drawdown [%]']), 'win_rate': float(stats['Win Rate [%]']),
                      'total_trades': int(stats['# Trades']), 'parameters': stats['_strategy'].__dict__}
    else:
        stats_dict = {'strategy_name': 'asia_liquidity_grab_reversal (Optimized)', 'return': 0, 'sharpe': 0,
                      'max_drawdown': 0, 'win_rate': 0, 'total_trades': 0,
                      'note': 'No trades executed during optimization.'}

    with open('results/temp_result.json', 'w') as f:
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating, np.bool_)): return obj.item()
                elif isinstance(obj, np.ndarray): return obj.tolist()
                elif hasattr(obj, 'name'): return obj.name
                return super(CustomEncoder, self).default(obj)
        json.dump(stats_dict, f, indent=2, cls=CustomEncoder)

    print("Backtest optimization stats saved to results/temp_result.json")
    bt.plot()
