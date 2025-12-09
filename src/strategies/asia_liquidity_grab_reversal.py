from backtesting import Backtest, Strategy
import pandas as pd
import json

# --- Data Pre-processing Function ---
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data to add Asia session high and low.

    NOTE: This strategy is designed for 24-hour Forex data. Using it with
    market-hour data like GOOG will produce unrealistic results. The session
    times are hardcoded as proxies and may not align with actual market sessions.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        # GOOG data from backtesting.test is not a DatetimeIndex, so we convert it
        df.index = pd.to_datetime(df.index)

    # Define session times (in UTC for real Forex data)
    # Using proxy times for GOOG data's timezone
    asia_session_start = pd.to_datetime('00:00').time()
    asia_session_end = pd.to_datetime('08:00').time()
    uk_session_start = pd.to_datetime('08:00').time()
    uk_session_end = pd.to_datetime('16:00').time()

    # Mark rows belonging to each session
    df['is_asia_session'] = (df.index.time >= asia_session_start) & (df.index.time < asia_session_end)
    df['is_uk_session'] = (df.index.time >= uk_session_start) & (df.index.time < uk_session_end)

    # Calculate daily Asia session high and low
    asia_session_data = df[df['is_asia_session']]
    daily_asia_high = asia_session_data.groupby(asia_session_data.index.date)['High'].max()
    daily_asia_low = asia_session_data.groupby(asia_session_data.index.date)['Low'].min()

    # Map daily values to the dataframe
    df['asia_high'] = pd.Series(df.index.date, index=df.index).map(daily_asia_high)
    df['asia_low'] = pd.Series(df.index.date, index=df.index).map(daily_asia_low)

    # Forward-fill the session levels to make them available throughout the day
    df['asia_high'] = df['asia_high'].ffill()
    df['asia_low'] = df['asia_low'].ffill()

    # Calculate Asia range and percentage
    df['asia_range'] = df['asia_high'] - df['asia_low']
    df['asia_range_pct'] = (df['asia_range'] / df['asia_low']) * 100

    # Clean up intermediate columns and drop rows with NaN values
    df = df.drop(columns=['is_asia_session'])
    df = df.dropna()

    return df

class AsiaLiquidityGrabReversalStrategy(Strategy):
    # Parameters for optimization
    asia_range_max_pct = 2.0

    def init(self):
        # The core logic will use the pre-calculated columns from the dataframe.
        # No need to use self.I() for these session levels.
        pass

    def next(self):
        # --- Strategy Filters ---
        # 1. Trade only during the UK session
        if not self.data.is_uk_session[-1]:
            return

        # 2. Asia session range filter
        if self.data.asia_range_pct[-1] > self.asia_range_max_pct:
            return

        # --- Reversal Candle Patterns ---
        # Bearish Engulfing: Open > Prev. Close AND Close < Prev. Open
        is_bearish_engulfing = self.data.Open[-1] > self.data.Close[-2] and self.data.Close[-1] < self.data.Open[-2]
        # Bullish Engulfing: Open < Prev. Close AND Close > Prev. Open
        is_bullish_engulfing = self.data.Open[-1] < self.data.Close[-2] and self.data.Close[-1] > self.data.Open[-2]

        # --- Entry Logic ---
        # Short Entry: False breakout above Asia High
        if self.data.High[-2] > self.data.asia_high[-1] and is_bearish_engulfing:
            if not self.position:
                sl = self.data.High[-1] # SL above the reversal candle high
                tp = self.data.asia_low[-1] # TP at the opposite side of the range
                self.sell(sl=sl, tp=tp)

        # Long Entry: False breakout below Asia Low
        elif self.data.Low[-2] < self.data.asia_low[-1] and is_bullish_engulfing:
            if not self.position:
                sl = self.data.Low[-1] # SL below the reversal candle low
                tp = self.data.asia_high[-1] # TP at the opposite side of the range
                self.buy(sl=sl, tp=tp)


if __name__ == '__main__':
    from backtesting.test import GOOG

    # --- Load and Preprocess Data ---
    # Using GOOG data as a placeholder. For real use, replace with 15M Forex data.
    # The GOOG dataset is not ideal as it has large gaps and no 24h data.
    data = GOOG.copy()
    data = data.iloc[::4] # Resample to ~15 min interval for faster test, not ideal
    data = preprocess_data(data)

    # --- Run Backtest ---
    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=10000, commission=.002)

    # --- Optimize ---
    stats = bt.optimize(
        asia_range_max_pct=range(1, 4, 1),
        maximize='Sharpe Ratio'
    )

    # --- Save Results ---
    import os
    os.makedirs('results', exist_ok=True)

    # Handle cases where no trades are made to avoid NaN in JSON
    if stats['# Trades'] > 0:
        results_dict = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }
    else:
        results_dict = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': 0.0,
            'sharpe': None,
            'max_drawdown': 0.0,
            'win_rate': None,
            'total_trades': 0
        }

    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    # --- Generate Plot ---
    bt.plot()
