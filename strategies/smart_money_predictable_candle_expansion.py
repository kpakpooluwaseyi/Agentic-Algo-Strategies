import json
import os
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

def preprocess_data(df):
    """
    Adds higher timeframe key levels to the DataFrame.
    """
    # Ensure index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # --- Calculate Daily/Weekly Levels ---
    df['day'] = df.index.date
    df['week_str'] = df.index.isocalendar().year.astype(str) + '_' + df.index.isocalendar().week.astype(str)

    daily_high_map = df.groupby('day')['High'].max().shift(1)
    daily_low_map = df.groupby('day')['Low'].min().shift(1)
    df['Prev_Day_High'] = df['day'].map(daily_high_map)
    df['Prev_Day_Low'] = df['day'].map(daily_low_map)

    # clean up helper columns
    df.drop(columns=['day', 'week_str'], inplace=True, errors='ignore')
    return df


class SmartMoneyPredictableCandleExpansion(Strategy):
    """
    Implements the Smart Money Predictable Candle Expansion strategy.
    """
    min_rr = 3.0
    sl_buffer_pct = 0.01
    displacement_body_pct = 0.5 # Body must be at least 50% of the candle's range

    confirmation_window = 5
    peak_prominence = 30

    def init(self):
        self.trade_state = 'WAITING_FOR_SWEEP'
        self.fvg_high = None
        self.fvg_low = None
        self.sl_price = None
        self.sweep_bar_index = 0

        # Pre-calculate swing points
        highs = self.data.High
        lows = self.data.Low

        # Custom indicator using self.I to run a function on the whole dataset
        def find_all_peaks(data):
            peaks, _ = find_peaks(data, prominence=self.peak_prominence)
            peak_indices = np.zeros_like(data)
            peak_indices[peaks] = 1
            return peak_indices

        self.swing_highs = self.I(find_all_peaks, highs, name="SwingHighs")
        self.swing_lows = self.I(find_all_peaks, -lows, name="SwingLows")


    def find_fvg_after_displacement(self):
        if len(self.data) < 3: return None, None, None

        is_bearish_displacement = (self.data.Close[-2] < self.data.Open[-2] and
                                   (self.data.Open[-2] - self.data.Close[-2]) / (self.data.High[-2] - self.data.Low[-2] + 1e-9) >= self.displacement_body_pct)
        if is_bearish_displacement and self.data.Low[-3] > self.data.High[-1]:
            return 'bearish', self.data.High[-1], self.data.Low[-3]

        is_bullish_displacement = (self.data.Close[-2] > self.data.Open[-2] and
                                   (self.data.Close[-2] - self.data.Open[-2]) / (self.data.High[-2] - self.data.Low[-2] + 1e-9) >= self.displacement_body_pct)
        if is_bullish_displacement and self.data.High[-3] < self.data.Low[-1]:
            return 'bullish', self.data.High[-3], self.data.Low[-1]

        return None, None, None

    def next(self):
        current_bar = len(self.data.Close) - 1

        if self.position: return

        if self.trade_state == 'WAITING_FOR_SWEEP':
            pdh, pdl = self.data.Prev_Day_High[-1], self.data.Prev_Day_Low[-1]
            if pd.isna(pdh) or pd.isna(pdl): return

            if self.data.High[-1] > pdh:
                self.trade_state = 'CONFIRMING_BEARISH_SETUP'
                self.sl_price, self.sweep_bar_index = self.data.High[-1], current_bar
            elif self.data.Low[-1] < pdl:
                self.trade_state = 'CONFIRMING_BULLISH_SETUP'
                self.sl_price, self.sweep_bar_index = self.data.Low[-1], current_bar

        elif self.trade_state in ('CONFIRMING_BEARISH_SETUP', 'CONFIRMING_BULLISH_SETUP'):
            if current_bar > self.sweep_bar_index + self.confirmation_window:
                self.trade_state = 'WAITING_FOR_SWEEP'
                return

            self.sl_price = max(self.sl_price, self.data.High[-1]) if self.trade_state == 'CONFIRMING_BEARISH_SETUP' else min(self.sl_price, self.data.Low[-1])

            fvg_dir, fvg_low, fvg_high = self.find_fvg_after_displacement()
            if fvg_dir and fvg_dir.startswith(self.trade_state.split('_')[1].lower()):
                self.fvg_low, self.fvg_high = fvg_low, fvg_high
                self.trade_state = f'WAITING_FOR_RETEST_{fvg_dir.upper()}'

        elif 'WAITING_FOR_RETEST' in self.trade_state:
            # --- Dealing Range Filter ---
            swing_high_indices = np.where(self.swing_highs[:current_bar])[0]
            swing_low_indices = np.where(self.swing_lows[:current_bar])[0]
            if len(swing_high_indices) == 0 or len(swing_low_indices) == 0:
                self.trade_state = 'WAITING_FOR_SWEEP'
                return

            last_swing_high_idx = swing_high_indices[-1]
            last_swing_low_idx = swing_low_indices[-1]

            range_high = self.data.High[last_swing_high_idx]
            range_low = self.data.Low[last_swing_low_idx]
            equilibrium = (range_high + range_low) / 2

            # --- Entry Logic ---
            if self.trade_state == 'WAITING_FOR_RETEST_BEARISH':
                if self.data.High[-1] > self.sl_price: self.trade_state = 'WAITING_FOR_SWEEP'
                elif self.data.High[-1] >= self.fvg_low and self.fvg_low > equilibrium:
                    entry_price = self.fvg_low
                    sl = self.sl_price * (1 + self.sl_buffer_pct)
                    risk = sl - entry_price

                    # Dynamic TP: Target the last swing low
                    tp_price = self.data.Low[last_swing_low_idx]
                    reward = entry_price - tp_price

                    if risk > 0 and (reward / risk) >= self.min_rr:
                        self.sell(limit=entry_price, sl=sl, tp=tp_price, size=0.1)
                    self.trade_state = 'WAITING_FOR_SWEEP'

            elif self.trade_state == 'WAITING_FOR_RETEST_BULLISH':
                if self.data.Low[-1] < self.sl_price: self.trade_state = 'WAITING_FOR_SWEEP'
                elif self.data.Low[-1] <= self.fvg_high and self.fvg_high < equilibrium:
                    entry_price = self.fvg_high
                    sl = self.sl_price * (1 - self.sl_buffer_pct)
                    risk = entry_price - sl

                    # Dynamic TP: Target the last swing high
                    tp_price = self.data.High[last_swing_high_idx]
                    reward = tp_price - entry_price

                    if risk > 0 and (reward / risk) >= self.min_rr:
                        self.buy(limit=entry_price, sl=sl, tp=tp_price, size=0.1)
                    self.trade_state = 'WAITING_FOR_SWEEP'

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Clean up column names: strip whitespace and capitalize
        data.columns = [c.strip().title() for c in data.columns]
    else:
        print("Data file not found. Please place BTC-USD-15m.csv in the 'data' directory.")
        exit()

    # Preprocess the data
    data = preprocess_data(data)

    # Run backtest
    bt = Backtest(data, SmartMoneyPredictableCandleExpansion, cash=100_000, commission=.002)

    print("Running single backtest with defaults...")
    stats = bt.run()
    print(stats)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save results to JSON, sanitizing for non-serializable types
    result = {
        'strategy_name': 'smart_money_predictable_candle_expansion',
        'return': stats.get('Return [%]', None),
        'sharpe': stats.get('Sharpe Ratio', None),
        'max_drawdown': stats.get('Max. Drawdown [%]', None),
        'win_rate': stats.get('Win Rate [%]', None),
        'total_trades': stats.get('# Trades', 0)
    }

    # Clean up NaN and numpy types
    for key, value in result.items():
        if pd.isna(value):
            result[key] = None
        elif isinstance(value, (np.int64, np.int32)):
            result[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            result[key] = float(value)

    with open('results/temp_result.json', 'w') as f:
        json.dump(result, f, indent=2)
        f.write('\n') # Add newline at the end

    # Generate plot
    plot_filename = 'results/smart_money_predictable_candle_expansion.html'
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
