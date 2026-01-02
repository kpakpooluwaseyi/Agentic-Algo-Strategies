
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


# Proxy Indicators for Market Cipher B
def money_flow_index(high, low, close, volume, length=14):
    """Proxy for Market Cipher B Money Flow using MFI."""
    mfi = ta.mfi(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), volume=pd.Series(volume), length=length)
    return mfi.values

def wavetrend_oscillator(high, low, close, channel_len=10, avg_len=21):
    """Proxy for Market Cipher B Momentum Waves using a WaveTrend Oscillator."""
    ap = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    esa = ap.ewm(span=channel_len, adjust=False).mean()
    d = (abs(ap - esa)).ewm(span=channel_len, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    wt1 = ci.ewm(span=avg_len, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    return wt1.values, wt2.values

def macd_indicator(close, fast=12, slow=26, signal=9):
    """Proxy for Market Cipher B MACD."""
    macd_df = ta.macd(close=pd.Series(close), fast=fast, slow=slow, signal=signal)
    macd_line = macd_df[f'MACD_{fast}_{slow}_{signal}']
    signal_line = macd_df[f'MACDs_{fast}_{slow}_{signal}']
    return macd_line.values, signal_line.values


class MarketCipherBTripleConfirmation(Strategy):
    # Indicator Parameters
    mfi_length = 14
    wt_channel_len = 10
    wt_avg_len = 21
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # Risk Management Parameters
    risk_reward_ratio = 2.0
    stop_loss_swing = 10  # Lookback period for swing high/low

    def init(self):
        # Initialize Indicators
        self.mfi = self.I(money_flow_index, self.data.High, self.data.Low, self.data.Close, self.data.Volume, length=self.mfi_length)
        self.wt1, self.wt2 = self.I(wavetrend_oscillator, self.data.High, self.data.Low, self.data.Close, channel_len=self.wt_channel_len, avg_len=self.wt_avg_len)
        self.macd, self.macd_signal = self.I(macd_indicator, self.data.Close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)

    def next(self):
        # Exit logic based on indicator reversal
        if self.position:
            if self.position.is_long:
                if self.mfi[-1] < self.mfi[-2] or self.wt1[-1] < self.wt2[-1] or self.macd[-1] < self.macd_signal[-1]:
                    self.position.close()
            elif self.position.is_short:
                if self.mfi[-1] > self.mfi[-2] or self.wt1[-1] > self.wt2[-1] or self.macd[-1] > self.macd_signal[-1]:
                    self.position.close()

        # Entry Conditions
        if not self.position:
            # Long Entry
            money_flow_bullish = self.mfi[-1] > self.mfi[-2]
            momentum_bullish = self.wt1[-1] > self.wt2[-1]
            macd_bullish = self.macd[-1] > self.macd_signal[-1]

            if money_flow_bullish and momentum_bullish and macd_bullish:
                swing_low = self.data.Low[-self.stop_loss_swing:].min()
                stop_loss = swing_low
                take_profit = self.data.Close[-1] + (self.data.Close[-1] - stop_loss) * self.risk_reward_ratio
                self.buy(sl=stop_loss, tp=take_profit)

            # Short Entry
            money_flow_bearish = self.mfi[-1] < self.mfi[-2]
            momentum_bearish = self.wt1[-1] < self.wt2[-1]
            macd_bearish = self.macd[-1] < self.macd_signal[-1]

            if money_flow_bearish and momentum_bearish and macd_bearish:
                swing_high = self.data.High[-self.stop_loss_swing:].max()
                stop_loss = swing_high
                take_profit = self.data.Close[-1] - (stop_loss - self.data.Close[-1]) * self.risk_reward_ratio
                self.sell(sl=stop_loss, tp=take_profit)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    def generate_synthetic_data():
        """Generates synthetic data for testing the strategy."""
        n_points = 5000
        index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))
        price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.1)
        price += np.sin(np.linspace(0, 200, n_points)) * 2
        data = pd.DataFrame({
            'Open': price, 'High': price * 1.005, 'Low': price * 0.995,
            'Close': price, 'Volume': np.random.randint(100, 1000, n_points)
        }, index=index)
        return data

    if os.path.exists(data_path):
        try:
            data = pd.read_csv(
                data_path, index_col='datetime', parse_dates=True,
            )
            data.columns = [col.strip().capitalize() for col in data.columns]
            # Ensure column names are capitalized correctly for backtesting.py
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'. Generating synthetic data.")
        data = generate_synthetic_data()

    data.dropna(inplace=True)
    bt = Backtest(data, MarketCipherBTripleConfirmation, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)): continue
            if isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, int): sanitized[key] = int(value)
            elif isinstance(value, pd.Timestamp): sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta): sanitized[key] = str(value)
            elif pd.isna(value): sanitized[key] = None
            elif key.startswith('_'): continue
            else: sanitized[key] = value
        return sanitized

    final_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    try:
        plot_filename = 'results/market_cipher_b_triple_confirmation.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
