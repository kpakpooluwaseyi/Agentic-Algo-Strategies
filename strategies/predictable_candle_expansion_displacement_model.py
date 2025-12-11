from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

def fvg_detector(df):
    """
    Vectorized Fair Value Gap (FVG) detector without lookahead bias.
    A bearish FVG is confirmed on the close of the third candle in a 3-bar pattern.
    The gap exists between the low of the first candle and the high of the third.
    """
    high = df.High
    low = df.Low

    # A bearish FVG is a 3-bar pattern. We look at t-2, t-1, and t (current bar).
    # The signal is confirmed at the close of t.
    first_candle_low = low.shift(2)
    third_candle_high = high.shift(0) # current bar's high

    # Condition: The high of the 3rd candle is below the low of the 1st candle.
    bearish_fvg_condition = third_candle_high < first_candle_low

    bearish_fvg_high = np.where(bearish_fvg_condition, first_candle_low, np.nan)
    bearish_fvg_low = np.where(bearish_fvg_condition, third_candle_high, np.nan)

    # Bullish FVG logic would be the inverse
    first_candle_high = high.shift(2)
    third_candle_low = low.shift(0)
    bullish_fvg_condition = third_candle_low > first_candle_high
    bullish_fvg_high = np.where(bullish_fvg_condition, third_candle_low, np.nan)
    bullish_fvg_low = np.where(bullish_fvg_condition, first_candle_high, np.nan)

    return bearish_fvg_high, bearish_fvg_low, bullish_fvg_high, bullish_fvg_low

def ob_detector(df):
    """
    Vectorized Order Block (OB) detector without lookahead bias.
    A bearish OB is the last up-candle before a strong down-move.
    The signal is confirmed after the down-move has occurred.
    """
    close = df.Close
    high = df.High
    low = df.Low

    # Bearish OB: A previous up-candle (t-1) followed by a strong down-candle (t)
    is_prev_up_candle = close.shift(1) > close.shift(2)
    is_curr_down_candle = close < close.shift(1)
    is_strong_move = close < low.shift(1) # Current close breaks the low of the previous candle

    bearish_ob_condition = is_prev_up_candle & is_curr_down_candle & is_strong_move
    bearish_ob_high = np.where(bearish_ob_condition, high.shift(1), np.nan)
    bearish_ob_low = np.where(bearish_ob_condition, low.shift(1), np.nan)

    # Bullish OB: A previous down-candle (t-1) followed by a strong up-candle (t)
    is_prev_down_candle = close.shift(1) < close.shift(2)
    is_curr_up_candle = close > close.shift(1)
    is_strong_up_move = close > high.shift(1)

    bullish_ob_condition = is_prev_down_candle & is_curr_up_candle & is_strong_up_move
    bullish_ob_high = np.where(bullish_ob_condition, high.shift(1), np.nan)
    bullish_ob_low = np.where(bullish_ob_condition, low.shift(1), np.nan)

    return bearish_ob_high, bearish_ob_low, bullish_ob_high, bullish_ob_low


def swing_highs(high, lookback):
    """
    Identifies swing highs using a rolling window without lookahead bias.
    """
    return high.shift(1).rolling(window=lookback).max()


class PredictableCandleExpansionDisplacementModelStrategy(Strategy):
    """
    Strategy based on identifying stop runs, displacements, and retests of FVGs or OBs.
    This implementation focuses on the SELL setup.
    """
    # Optimizable parameters
    rr = 3  # Risk/Reward Ratio
    stop_run_lookback = 20 # Lookback period for identifying swing highs for stop runs
    invalidation_period = 10 # Bars to wait for a retest before invalidating the setup

    def init(self):
        # Pre-calculate indicators and signals
        self.daily_high = self.I(lambda x: pd.Series(x).resample('D').max().ffill(), self.data.df.High)
        self.daily_low = self.I(lambda x: pd.Series(x).resample('D').min().ffill(), self.data.df.Low)
        self.swing_high = self.I(swing_highs, self.data.df.High, self.stop_run_lookback)

        self.bearish_fvg_high, self.bearish_fvg_low, _, _ = self.I(fvg_detector, self.data.df)
        self.bearish_ob_high, self.bearish_ob_low, _, _ = self.I(ob_detector, self.data.df)

        # State variables
        self.stop_run_high = None
        self.displacement_fvg_range = None
        self.displacement_ob_range = None
        self.stop_loss_price = None
        self.setup_bar_index = 0

    def next(self):
        # --- STATE INVALIDATION ---
        if self.stop_run_high and (len(self.data) - self.setup_bar_index > self.invalidation_period):
            self.stop_run_high = None
            self.displacement_fvg_range = None
            self.displacement_ob_range = None

        # --- SELL SETUP ---

        # 1. Stop Run: Price trades above PDH or a recent swing high
        pdh_run = self.data.High[-1] > self.daily_high[-1]
        swing_high_run = self.data.High[-1] > self.swing_high[-1]

        is_stop_run_active = self.stop_run_high is not None

        # Initial trigger for the stop run
        if not is_stop_run_active and (pdh_run or swing_high_run):
            self.stop_run_high = self.data.High[-1]
            self.stop_loss_price = self.stop_run_high
            self.setup_bar_index = len(self.data)
            is_stop_run_active = True

        # If a stop run is active, check for displacement or new highs
        if is_stop_run_active:
            # Check for a displacement signal (FVG or OB)
            has_fvg = not np.isnan(self.bearish_fvg_high[-1])
            has_ob = not np.isnan(self.bearish_ob_high[-1])

            if has_fvg or has_ob:
                if has_fvg:
                    self.displacement_fvg_range = (self.bearish_fvg_low[-1], self.bearish_fvg_high[-1])
                else:
                    self.displacement_ob_range = (self.bearish_ob_low[-1], self.bearish_ob_high[-1])
                self.setup_bar_index = len(self.data)

            # If no displacement yet, check if price is making a new high
            elif self.data.High[-1] > self.stop_run_high:
                self.stop_run_high = self.data.High[-1]
                self.stop_loss_price = self.stop_run_high
                self.setup_bar_index = len(self.data) # Reset invalidation timer

        # 3. Entry: Price retests the FVG or OB
        if self.position.is_short:
            return

        entry_price = None

        # Retest of FVG
        if self.displacement_fvg_range:
            fvg_low, fvg_high = self.displacement_fvg_range
            if self.data.High[-1] >= fvg_low and self.data.Low[-1] <= fvg_high:
                entry_price = (fvg_high + fvg_low) / 2 # Enter at the midpoint of the FVG

        # Retest of OB
        elif self.displacement_ob_range:
            ob_low, ob_high = self.displacement_ob_range
            if self.data.High[-1] >= ob_low and self.data.Low[-1] <= ob_high:
                entry_price = (ob_high + ob_low) / 2 # Enter at the midpoint of the OB

        if entry_price:
            sl = self.stop_loss_price
            tp = entry_price - (sl - entry_price) * self.rr

            if tp < entry_price:
                self.sell(limit=entry_price, sl=sl, tp=tp)

                # Reset state after entry
                self.stop_run_high = None
                self.displacement_fvg_range = None
                self.displacement_ob_range = None
                self.stop_loss_price = None

if __name__ == '__main__':
    # Generate synthetic data for a textbook sell setup
    def generate_synthetic_data():
        t = pd.date_range(start='2023-01-01', periods=200, freq='15min')
        price = 100 + np.random.randn(200).cumsum() * 0.1
        data = pd.DataFrame({'Open': price, 'High': price, 'Low': price, 'Close': price,
                             'Volume': np.random.randint(100, 1000, 200)}, index=t)

        # Day 1: Establish a clear high
        data.iloc[20:40, 2] = 98 # Low
        data.iloc[30, 1] = 105  # Previous Day High (PDH)

        # Day 2: The setup
        # 1. Multi-bar Stop Run above PDH
        data.iloc[97, 1] = 105.5 # Initial break of PDH
        data.iloc[98, 1] = 106.0 # The true peak of the stop run
        data.iloc[98, 3] = 105.8 # Close

        # 2. Displacement creating an FVG (3-bar pattern)
        # Bar 1 of FVG pattern
        data.iloc[99, 2] = 105.0
        # Bar 2 of FVG pattern (the gap)
        data.iloc[100, 1] = 104.5
        data.iloc[100, 2] = 103.5
        # Bar 3 of FVG pattern (confirms the gap)
        data.iloc[101, 1] = 103.0 # This high is below Bar 1's low, confirming the FVG

        # 3. Retest of the FVG, ensuring the limit order price is touched
        data.iloc[102, 1] = 104.1 # High
        data.iloc[102, 2] = 103.9 # Low - ensuring the midpoint of the FVG (104.0) is crossed

        # Ensure OHLC consistency
        data['Open'] = data['Close'].shift(1).fillna(data['Close'])
        data.loc[data['High'] < data['Open'], 'Open'] = data['High']
        data.loc[data['High'] < data['Close'], 'Close'] = data['High']
        data.loc[data['Low'] > data['Open'], 'Open'] = data['Low']
        data.loc[data['Low'] > data['Close'], 'Close'] = data['Low']

        return data

    data = generate_synthetic_data()

    # Run backtest
    bt = Backtest(data, PredictableCandleExpansionDisplacementModelStrategy, cash=100000, commission=.002)

    # Optimize
    stats = bt.optimize(
        rr=np.arange(1.5, 4, 0.5),
        stop_run_lookback=range(10, 31, 5),
        invalidation_period=range(5, 16, 5),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.rr > 1
    )

    print(stats)

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON serialization
    sanitized_stats = {key: (value if pd.isna(value) else (int(value) if isinstance(value, np.integer) else float(value))) for key, value in stats.items() if isinstance(value, (np.integer, np.floating, float, int))}
    sanitized_stats['strategy_name'] = 'predictable_candle_expansion_displacement_model'

    # Add required fields that might be missing from optimization stats
    final_results = {
        'strategy_name': sanitized_stats.get('strategy_name'),
        'return': sanitized_stats.get('Return [%]'),
        'sharpe': sanitized_stats.get('Sharpe Ratio'),
        'max_drawdown': sanitized_stats.get('Max. Drawdown [%]'),
        'win_rate': sanitized_stats.get('Win Rate [%]'),
        'total_trades': sanitized_stats.get('# Trades')
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # Generate plot
    try:
        bt.plot(filename='results/plot.html')
    except TypeError as e:
        print(f"Could not generate plot due to a known issue with the plotting library: {e}")
