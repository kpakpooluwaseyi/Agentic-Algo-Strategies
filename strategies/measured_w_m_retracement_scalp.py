from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks

# ============== Strategy Logic ==============

class MeasuredWMRetracementScalpStrategy(Strategy):
    # Define optimization parameters
    aoi_padding_pips = 0.0005
    reversal_confirmation_candles = 3
    min_rr = 2.0
    confluence_pips = 0.0010 # 10 pips for confluence check

    def init(self):
        # State variables for tracking setups
        self.active_setup_id = None
        self.invalidation_breached = False
        self.aoi_armed = False

    def next(self):
        current_index = len(self.data.Close) - 1
        current_setup_id = self.data.df['setup_id'].iloc[current_index]

        # --- State Management ---
        # If a new setup is detected, reset state
        if current_setup_id != self.active_setup_id and not pd.isna(current_setup_id):
            self.active_setup_id = current_setup_id
            self.invalidation_breached = False
            self.aoi_armed = False

        # If there is no active setup, do nothing
        if self.active_setup_id is None:
            return

        # If the setup has already been invalidated, do nothing
        if self.invalidation_breached:
            return

        # --- Fibonacci Level and Confluence Check ---
        setup = self.data.df.iloc[current_index]
        price_high = self.data.High[-1]
        price_low = self.data.Low[-1]

        # Check for invalidation or arming only if not already armed
        if not self.aoi_armed:
            if setup['setup_direction'] == -1: # M-Setup (Short)
                if price_high > setup['fib_38'] or price_high > setup['fib_61'] or price_high > setup['fib_78']:
                    self.invalidation_breached = True
                    return
                if price_high >= setup['fib_50']:
                    # Check for confluence with key levels
                    if abs(setup['fib_50'] - setup['asia_high']) < self.confluence_pips or \
                       abs(setup['fib_50'] - setup['prev_day_high']) < self.confluence_pips:
                        self.aoi_armed = True

            elif setup['setup_direction'] == 1: # W-Setup (Long)
                if price_low < setup['fib_38'] or price_low < setup['fib_61'] or price_low < setup['fib_78']:
                    self.invalidation_breached = True
                    return
                if price_low <= setup['fib_50']:
                    if abs(setup['fib_50'] - setup['asia_low']) < self.confluence_pips or \
                       abs(setup['fib_50'] - setup['prev_day_low']) < self.confluence_pips:
                        self.aoi_armed = True

        # --- Entry Logic ---
        if self.aoi_armed:
            if setup['setup_direction'] == -1: # SHORT
                if len(self.data.Close) < self.reversal_confirmation_candles + 1: return
                recent_highs = self.data.High[-self.reversal_confirmation_candles:]
                center_peak = max(recent_highs)
                if self.data.Close[-1] < self.data.Close[-2] and self.data.High[-2] == center_peak:
                    retrace_low = min(self.data.Low[-self.reversal_confirmation_candles:])
                    tp = center_peak - (center_peak - retrace_low) * 0.5
                    sl = center_peak + self.aoi_padding_pips
                    risk = abs(self.data.Close[-1] - sl)
                    reward = abs(self.data.Close[-1] - tp)
                    if risk > 0 and reward / risk >= self.min_rr and sl > self.data.Close[-1] and tp < self.data.Close[-1]:
                        self.sell(sl=sl, tp=tp)
                        self.aoi_armed = False # Reset after entry

            elif setup['setup_direction'] == 1: # LONG
                if len(self.data.Close) < self.reversal_confirmation_candles + 1: return
                recent_lows = self.data.Low[-self.reversal_confirmation_candles:]
                center_trough = min(recent_lows)
                if self.data.Close[-1] > self.data.Close[-2] and self.data.Low[-2] == center_trough:
                    retrace_high = max(self.data.High[-self.reversal_confirmation_candles:])
                    tp = center_trough + (retrace_high - center_trough) * 0.5
                    sl = center_trough - self.aoi_padding_pips
                    risk = abs(self.data.Close[-1] - sl)
                    reward = abs(self.data.Close[-1] - tp)
                    if risk > 0 and reward / risk >= self.min_rr and sl < self.data.Close[-1] and tp > self.data.Close[-1]:
                        self.buy(sl=sl, tp=tp)
                        self.aoi_armed = False # Reset after entry


# ============== Data Pre-processing ==============

def generate_forex_data(days=90):
    rng = np.random.default_rng(42)
    n_points = days * 24 * 60
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='min', tz='UTC'))
    price = 1.0700 + np.sin(np.arange(n_points) * np.pi / (12 * 60)) * 0.005
    price += np.cumsum(rng.normal(0, 0.00001, n_points))
    volatility = np.ones(n_points) * 0.0001
    for i in range(days):
        volatility[i*1440 + 8*60 : i*1440 + 16*60] *= 2
    price += rng.normal(0, volatility, n_points)
    df = pd.DataFrame(index=index)
    df['Open'] = price
    df['High'] = df['Open'] + rng.uniform(0, 0.0003, n_points)
    df['Low'] = df['Open'] - rng.uniform(0, 0.0003, n_points)
    df['Close'] = df['Open'] + rng.normal(0, 0.00015, n_points)
    df['Volume'] = rng.integers(100, 1000, size=n_points)
    return df

def find_swing_points(series, order=5):
    peaks_idx, _ = find_peaks(series, distance=order, width=order)
    troughs_idx, _ = find_peaks(-series, distance=order, width=order)
    return peaks_idx, troughs_idx

def preprocess_data(data_1m):
    # --- Calculate Key Price Levels ---
    data_1m['date'] = data_1m.index.date
    daily_data = data_1m.resample('D').agg({'High': 'max', 'Low': 'min'}).dropna()
    data_1m['prev_day_high'] = data_1m['date'].map(daily_data['High'].shift(1))
    data_1m['prev_day_low'] = data_1m['date'].map(daily_data['Low'].shift(1))

    asia_session = data_1m.between_time('00:00', '08:00').resample('D').agg({'High':'max', 'Low':'min'}).dropna()
    data_1m['asia_high'] = data_1m['date'].map(asia_session['High'])
    data_1m['asia_low'] = data_1m['date'].map(asia_session['Low'])
    data_1m.ffill(inplace=True)

    # --- Identify 15M Setups ---
    data_15m = data_1m.resample('15min').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    peaks_idx, troughs_idx = find_swing_points(data_15m['Close'], order=5)

    setups = []
    setup_counter = 1

    for i in range(len(peaks_idx) - 1):
        p1_idx, p2_idx = peaks_idx[i], peaks_idx[i+1]
        troughs_between = troughs_idx[(troughs_idx > p1_idx) & (troughs_idx < p2_idx)]
        if len(troughs_between) > 0:
            t_idx = troughs_between[0]
            major_high = max(data_15m['High'].iloc[p1_idx], data_15m['High'].iloc[p2_idx])
            major_low = data_15m['Low'].iloc[t_idx]
            diff = major_high - major_low
            setups.append({
                'timestamp': data_15m.index[p2_idx], 'setup_id': setup_counter, 'setup_direction': -1,
                'fib_38': major_high - diff * 0.382, 'fib_50': major_high - diff * 0.50,
                'fib_61': major_high - diff * 0.618, 'fib_78': major_high - diff * 0.786
            })
            setup_counter += 1

    for i in range(len(troughs_idx) - 1):
        t1_idx, t2_idx = troughs_idx[i], troughs_idx[i+1]
        peaks_between = peaks_idx[(peaks_idx > t1_idx) & (peaks_idx < t2_idx)]
        if len(peaks_between) > 0:
            p_idx = peaks_between[0]
            major_low = min(data_15m['Low'].iloc[t1_idx], data_15m['Low'].iloc[t2_idx])
            major_high = data_15m['High'].iloc[p_idx]
            diff = major_high - major_low
            setups.append({
                'timestamp': data_15m.index[t2_idx], 'setup_id': setup_counter, 'setup_direction': 1,
                'fib_38': major_low + diff * 0.382, 'fib_50': major_low + diff * 0.50,
                'fib_61': major_low + diff * 0.618, 'fib_78': major_low + diff * 0.786
            })
            setup_counter += 1

    if not setups: return data_1m.drop(columns=['date'])

    setup_df = pd.DataFrame(setups).set_index('timestamp')
    data_1m_with_setups = pd.merge_asof(
        data_1m.sort_index(), setup_df.sort_index(),
        left_index=True, right_index=True,
        direction='forward', tolerance=pd.Timedelta('120min') # Setup is valid for 2 hours
    )
    return data_1m_with_setups.drop(columns=['date'])

# ============== Backtest Execution ==============

if __name__ == '__main__':
    data_1m = generate_forex_data(days=90)
    processed_data = preprocess_data(data_1m)

    if processed_data is None or 'setup_id' not in processed_data.columns:
        print("No setups found in the data. Exiting.")
    else:
        bt = Backtest(processed_data, MeasuredWMRetracementScalpStrategy, cash=10000, commission=.0002)

        stats = bt.optimize(
            reversal_confirmation_candles=range(3, 8, 2),
            min_rr=[1.5, 2.0, 2.5],
            confluence_pips=[p*0.0001 for p in range(5, 16, 5)],
            maximize='Sharpe Ratio'
        )

        print(stats)

        import os
        os.makedirs('results', exist_ok=True)

        # Save results
        stats_dict = stats.to_dict()
        if stats['_trades'].shape[0] > 0:
            result_dict = {
                'strategy_name': 'measured_w_m_retracement_scalp',
                'return': float(stats_dict.get('Return [%]', 0.0)),
                'sharpe': float(stats_dict.get('Sharpe Ratio', 0.0)),
                'max_drawdown': float(stats_dict.get('Max. Drawdown [%]', 0.0)),
                'win_rate': float(stats_dict.get('Win Rate [%]', 0.0)),
                'total_trades': int(stats_dict.get('# Trades', 0))
            }
        else:
            result_dict = {'strategy_name':'measured_w_m_retracement_scalp','return':0.0,'sharpe':0.0,'max_drawdown':0.0,'win_rate':0.0,'total_trades':0}

        with open('results/temp_result.json', 'w') as f:
            json.dump(result_dict, f, indent=2)

        bt.plot()
