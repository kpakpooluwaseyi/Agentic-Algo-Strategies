
from backtesting import Strategy
from backtesting.lib import resample_apply
import numpy as np
import pandas as pd
import pandas_ta as ta
import json
import os

def TEMA(series, timeperiod):
    """Calculates the Triple Exponential Moving Average (TEMA)."""
    ema1 = pd.Series(series).ewm(span=timeperiod, adjust=False).mean()
    ema2 = ema1.ewm(span=timeperiod, adjust=False).mean()
    ema3 = ema2.ewm(span=timeperiod, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

class RelativeStrengthMeanReversion(Strategy):
    # --- Strategy Parameters ---
    long_term_ma_period = 200
    atr_period = 14
    overbought_threshold = 1.05
    oversold_threshold = 0.95
    stop_loss_atr_multiplier = 2.0
    take_profit_atr_multiplier = 3.0
    volume_confirmation_multiplier = 1.2

    def init(self):
        # --- Indicators ---
        self.long_term_ma = self.I(ta.sma, pd.Series(self.data.Close), length=self.long_term_ma_period)
        self.relative_strength = self.data.Close / self.long_term_ma
        self.atr = self.I(ta.atr, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), length=self.atr_period)
        self.avg_volume = self.I(ta.sma, pd.Series(self.data.Volume), length=self.long_term_ma_period)

        # --- Higher Timeframe Trend Filter (4H) ---
        self.higher_tf_tema = resample_apply('4H', TEMA, self.data.Close, timeperiod=50)

    def next(self):
        # --- Trend Filter ---
        is_uptrend = self.data.Close[-1] > self.higher_tf_tema[-1]
        is_downtrend = self.data.Close[-1] < self.higher_tf_tema[-1]

        # --- Volume Confirmation ---
        is_volume_confirmed = self.data.Volume[-1] > self.avg_volume[-1] * self.volume_confirmation_multiplier

        # --- Entry Conditions ---
        if not self.position:
            # Short Entry (Overbought)
            if self.relative_strength[-1] > self.overbought_threshold and is_downtrend and is_volume_confirmed:
                sl = self.data.Close[-1] + self.atr[-1] * self.stop_loss_atr_multiplier
                tp = self.data.Close[-1] - self.atr[-1] * self.take_profit_atr_multiplier
                self.sell(sl=sl, tp=tp)

            # Long Entry (Oversold)
            elif self.relative_strength[-1] < self.oversold_threshold and is_uptrend and is_volume_confirmed:
                sl = self.data.Close[-1] - self.atr[-1] * self.stop_loss_atr_multiplier
                tp = self.data.Close[-1] + self.atr[-1] * self.take_profit_atr_multiplier
                self.buy(sl=sl, tp=tp)

if __name__ == '__main__':
    from backtesting import Backtest

    os.makedirs('results', exist_ok=True)

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

    bt = Backtest(df, RelativeStrengthMeanReversion, cash=100_000, commission=.002)
    stats = bt.run()

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
        bt.plot(filename='results/relative_strength_mean_reversion.html', open_browser=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")
