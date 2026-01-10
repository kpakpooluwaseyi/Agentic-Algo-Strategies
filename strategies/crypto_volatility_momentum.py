"""
Crypto Volatility Momentum Strategy
"""

from backtesting import Strategy, Backtest
import pandas as pd
import pandas_ta as ta


# Helper functions to wrap pandas-ta indicators for backtesting.py
def ATR(high, low, close, length):
    """Calculate ATR and return values."""
    atr_series = ta.atr(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), length=length)
    return atr_series.values if atr_series is not None else [0] * len(high)

def SMA(series, length):
    """Calculate SMA and return values."""
    sma_series = ta.sma(close=pd.Series(series), length=length)
    return sma_series.values if sma_series is not None else [0] * len(series)

def MOM(series, length):
    """Calculate Momentum and return values."""
    mom_series = ta.mom(close=pd.Series(series), length=length)
    return mom_series.values if mom_series is not None else [0] * len(series)


class CryptoVolatilityMomentum(Strategy):
    """
    A strategy that enters trades based on volatility and momentum.
    It goes long when momentum is positive and volatility is high,
    and goes short when momentum is negative and volatility is high.
    Risk is managed with an ATR-based stop-loss and take-profit.
    """
    # Optimizable parameters
    atr_period = 14
    momentum_period = 10
    volatility_threshold_multiplier = 1.0  # Volatility is "high" if ATR > SMA(ATR) * multiplier
    atr_sma_period = 50
    sl_multiplier = 2.0  # Stop loss at 2 * ATR
    tp_multiplier = 3.0  # Take profit at 3 * ATR

    def init(self):
        """
        Initialize the indicators.
        """
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, length=self.atr_period)
        self.atr_sma = self.I(SMA, self.atr, length=self.atr_sma_period)
        self.momentum = self.I(MOM, self.data.Close, length=self.momentum_period)

    def next(self):
        """
        Define the trading logic.
        """
        # Wait for indicators to warm up
        if len(self.data) < self.atr_sma_period:
            return

        price = self.data.Close[-1]
        atr_value = self.atr[-1]

        # Check for high volatility condition
        is_high_volatility = atr_value > (self.atr_sma[-1] * self.volatility_threshold_multiplier)

        # Entry logic
        if not self.position and is_high_volatility:
            # Go long if momentum is positive
            if self.momentum[-1] > 0:
                sl = price - atr_value * self.sl_multiplier
                tp = price + atr_value * self.tp_multiplier
                self.buy(sl=sl, tp=tp)

            # Go short if momentum is negative
            elif self.momentum[-1] < 0:
                sl = price + atr_value * self.sl_multiplier
                tp = price - atr_value * self.tp_multiplier
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found.")
        # As a fallback, create some dummy data
        from backtesting.test import EURUSD as df
        df = df.iloc[-2000:]

    # Clean data
    df = df.iloc[:, :-1]  # Drop the last unnamed column from trailing comma
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    bt = Backtest(df, CryptoVolatilityMomentum, cash=100_000, commission=.002)
    stats = bt.run()
    print(stats)

    # Save the stats to a JSON file
    import json
    # Sanitize stats for JSON serialization
    sanitized_stats = {key: str(value) if isinstance(value, (pd.Timestamp, pd.Timedelta)) else value
                       for key, value in stats.items() if not isinstance(value, (pd.Series, pd.DataFrame))}
    sanitized_stats.pop('_strategy', None)
    sanitized_stats.pop('_equity_curve', None)
    sanitized_stats.pop('_trades', None)

    with open("results/temp_result.json", "w") as f:
        json.dump(sanitized_stats, f, indent=4)

    bt.plot(filename="results/crypto_volatility_momentum.html")
