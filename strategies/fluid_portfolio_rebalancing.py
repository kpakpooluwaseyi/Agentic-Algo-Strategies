"""
Strategy: Fluid Portfolio Rebalancing (Mean Reversion Proxy)
============================================================

This strategy is a mean-reversion implementation adapted to the project's
strict development guidelines, serving as a proxy for the original
"fluid portfolio rebalancing" concept.

Proxy Logic:
- "Normal Level": Proxied by a long-term Exponential Moving Average (EMA).
- "Buy Low": Enter a long position when the price drops a certain ATR multiple
  below the EMA, but only during a higher-timeframe uptrend.
- "Sell High": Enter a short position when the price rises a certain ATR
  multiple above the EMA, but only during a higher-timeframe downtrend.

This approach adheres to the systematic, rule-based nature of the original
request while complying with all mandatory development guidelines.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Strategy, Backtest


def preprocess_data(df: pd.DataFrame, **params):
    """
    Adds all required indicators to the DataFrame. This function is designed to
    be called by the backtesting framework.
    """
    # Get parameters or use defaults from the class
    ema_period = params.get('ema_period', 200)
    atr_period = params.get('atr_period', 14)
    atr_multiplier = params.get('atr_multiplier', 2.0)
    volume_ma_period = params.get('volume_ma_period', 20)

    # ATR for bands and risk management
    df.ta.atr(high='High', low='Low', close='Close', length=atr_period, append=True)
    atr_col = f'ATRr_{atr_period}'
    df.rename(columns={atr_col: 'atr'}, inplace=True)

    # Central EMA and Mean Reversion Bands
    ema_col = f'ema_{ema_period}'
    df[ema_col] = ta.ema(df['Close'], length=ema_period)
    df['upper_band'] = df[ema_col] + (df['atr'] * atr_multiplier)
    df['lower_band'] = df[ema_col] - (df['atr'] * atr_multiplier)

    # Higher Timeframe (HTF) Trend Filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    df_4h['htf_ema'] = ta.ema(df_4h['Close'], length=ema_period)
    df_4h['htf_trend'] = np.where(df_4h['Close'] > df_4h['htf_ema'], 1, -1)

    # Map trend back to original timeframe and fill initial NaNs
    df['htf_trend'] = df_4h['htf_trend'].reindex(df.index, method='ffill').fillna(0)

    # Volume MA for confirmation
    volume_ma_col = f'volume_ma_{volume_ma_period}'
    df[volume_ma_col] = ta.sma(df['Volume'], length=volume_ma_period)

    # Clean up NaNs created by indicators at the start of the data
    df.dropna(inplace=True)

    return df


class FluidPortfolioRebalancing(Strategy):
    """
    A mean-reversion strategy that trades pullbacks from a central EMA,
    filtered by higher-timeframe trend and volume confirmation, with ATR-based
    risk management.
    """

    # Optimizable Parameters
    ema_period = 200
    atr_period = 14
    atr_multiplier = 2.0
    volume_ma_period = 20

    sl_atr_multiplier = 2.0
    tp_atr_multiplier = 3.0

    def init(self):
        """
        Initialize indicators from the preprocessed DataFrame.
        """
        # Create aliases for the pre-calculated indicator columns
        self.ema = self.I(lambda: self.data.df[f'ema_{self.ema_period}'], name='EMA')
        self.upper_band = self.I(lambda: self.data.df['upper_band'], name='Upper Band')
        self.lower_band = self.I(lambda: self.data.df['lower_band'], name='Lower Band')
        self.atr = self.I(lambda: self.data.df['atr'], name='ATR')
        self.htf_trend = self.I(lambda: self.data.df['htf_trend'], name='HTF Trend')
        self.volume_ma = self.I(lambda: self.data.df[f'volume_ma_{self.volume_ma_period}'], name='Volume MA')

    def next(self):
        """
        Main trading logic. Called on each bar.
        """
        price = self.data.Close[-1]

        # Condition 1: Adhere to higher-timeframe trend
        is_uptrend = self.htf_trend[-1] == 1
        is_downtrend = self.htf_trend[-1] == -1

        # Condition 2: Volume confirmation
        volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]

        # If already in a position, do nothing.
        if self.position:
            return

        # Long Entry Logic
        if is_uptrend and volume_confirmed and self.data.Low[-1] < self.lower_band[-1]:
            # Set SL and TP based on ATR
            sl = price - (self.sl_atr_multiplier * self.atr[-1])
            # For longs, TP must be higher than entry. Central EMA is the minimum target.
            tp = max(price + (self.tp_atr_multiplier * self.atr[-1]), self.ema[-1])
            self.buy(sl=sl, tp=tp)

        # Short Entry Logic
        elif is_downtrend and volume_confirmed and self.data.High[-1] > self.upper_band[-1]:
            # Set SL and TP based on ATR
            sl = price + (self.sl_atr_multiplier * self.atr[-1])
            # For shorts, TP must be lower than entry. Central EMA is the minimum target.
            tp = min(price - (self.tp_atr_multiplier * self.atr[-1]), self.ema[-1])
            self.sell(sl=sl, tp=tp)

# Standalone execution block
if __name__ == '__main__':
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Data file not found. Please ensure 'data/BTC-USD-15m.csv' exists.")
        exit()

    # Preprocess data
    df_processed = preprocess_data(df.copy())

    # Backtest
    bt = Backtest(df_processed, FluidPortfolioRebalancing, cash=100_000, commission=.001)
    stats = bt.run()

    print(stats)

    # Save results
    import json
    with open('results/temp_result.json', 'w') as f:
        json.dump(stats.to_dict(), f, indent=4)

    bt.plot(filename='results/fluid_portfolio_rebalancing.html', open_browser=False)

    print("Backtest complete. Results saved to 'results/'.")
