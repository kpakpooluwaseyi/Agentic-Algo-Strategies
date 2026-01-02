
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta as ta
from scipy.signal import find_peaks
import numpy as np

def money_flow_indicator(open_p, high_p, low_p, close_p, period=60, mult=200, y=2.25):
    """
    Custom implementation of the 'Money Flow' indicator based on the
    publicly available Pine Script code for a Market Cipher B clone.
    This is not a standard MFI, but a custom momentum oscillator.
    """
    price_range = high_p - low_p
    # Avoid division by zero
    price_range[price_range == 0] = 1e-9

    raw_mf = ((close_p - open_p) / price_range) * mult
    # The original script seems to use a TEMA (Triple Exponential Moving Average)
    # or a similar smoothed average. We'll use EMA here for simplicity and performance.
    smoothed_mf = pd.Series(raw_mf).ewm(span=period, adjust=False).mean()

    return (smoothed_mf - y).values

class MarketCipherBMacdMoneyFlowCrossover(Strategy):
    """
    Strategy that uses a custom 'Money Flow' indicator (emulating Market Cipher B),
    MACD, and price action to identify trend-following entry opportunities.
    """
    # === Strategy Parameters ===
    mf_period = 60
    mf_mult = 200
    mf_y_offset = 2.25

    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    swing_lookback = 30
    risk_reward_ratio = 2.0

    def init(self):
        """
        Initialize indicators and strategy state.
        """
        # --- Custom Money Flow Indicator ---
        self.money_flow = self.I(money_flow_indicator,
                                 self.data.Open, self.data.High, self.data.Low, self.data.Close,
                                 period=self.mf_period, mult=self.mf_mult, y=self.mf_y_offset)

        # --- Standard MACD ---
        macd_df = ta.macd(pd.Series(self.data.Close), fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        self.macd = self.I(lambda: macd_df.iloc[:, 0], name='MACD')
        self.macd_signal = self.I(lambda: macd_df.iloc[:, 2], name='MACD_Signal')

    def next(self):
        """
        Define the trading logic for the next bar.
        """
        # === Find recent swing points for Stop Loss placement ===
        lookback_slice = self.data.df[-self.swing_lookback:]
        highs_indices, _ = find_peaks(lookback_slice['High'], distance=5)
        lows_indices, _ = find_peaks(-lookback_slice['Low'], distance=5)

        recent_swing_high = lookback_slice['High'].iloc[highs_indices].max() if len(highs_indices) > 0 else np.inf
        recent_swing_low = lookback_slice['Low'].iloc[lows_indices].min() if len(lows_indices) > 0 else -np.inf

        # === Entry Conditions ===
        price = self.data.Close[-1]

        # --- Long ---
        breakout_long = self.data.Close[-1] > recent_swing_high
        long_signal = (
            crossover(self.money_flow, 0) and
            crossover(self.macd, self.macd_signal) and
            breakout_long
        )

        # --- Short ---
        breakout_short = self.data.Close[-1] < recent_swing_low
        short_signal = (
            crossover(0, self.money_flow) and
            crossover(self.macd_signal, self.macd) and
            breakout_short
        )

        # === Execution ===
        if not self.position:
            if long_signal:
                sl = recent_swing_low
                tp = price + (price - sl) * self.risk_reward_ratio
                if sl < price and tp > price:
                    self.buy(sl=sl, tp=tp)

            elif short_signal:
                sl = recent_swing_high
                tp = price - (sl - price) * self.risk_reward_ratio
                if sl > price and tp < price:
                    self.sell(sl=sl, tp=tp)

        # === Exit Conditions (based on indicator reversal) ===
        if self.position.is_long and (crossover(0, self.money_flow) or crossover(self.macd_signal, self.macd)):
            self.position.close()

        if self.position.is_short and (crossover(self.money_flow, 0) or crossover(self.macd, self.macd_signal)):
            self.position.close()

def sanitize_stats_for_json(stats):
    if stats is None: return {}
    clean_stats = stats.to_dict()
    for key in ['_strategy', '_equity_curve', '_trades']:
        if key in clean_stats: del clean_stats[key]
    for key, value in clean_stats.items():
        if isinstance(value, pd.Timestamp): clean_stats[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta): clean_stats[key] = str(value)
        elif isinstance(value, (np.int64, np.int32, np.float64, np.float32)): clean_stats[key] = value.item()
    return clean_stats

if __name__ == '__main__':
    DATA_PATH = 'data/BTC-USD-15m.csv'
    try:
        data = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)
        data.columns = [col.strip().capitalize() for col in data.columns]
    except FileNotFoundError:
        print(f"Data file not found at {DATA_PATH}.")
        exit()

    bt = Backtest(data, MarketCipherBMacdMoneyFlowCrossover, cash=100_000, commission=.002)
    stats = bt.run()
    print(stats)

    try:
        plot_filename = f"results/{MarketCipherBMacdMoneyFlowCrossover.__name__}.html"
        bt.plot(filename=plot_filename, open_browser=False)
    except Exception as e:
        print(f"Error plotting: {e}")

    import json
    results_dict = sanitize_stats_for_json(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    print(f"\nResults saved to results/temp_result.json")
    print(f"Plot saved to {plot_filename}")
