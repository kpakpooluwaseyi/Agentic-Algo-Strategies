
import json
import os
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.test import GOOG
from scipy.signal import find_peaks

# --- Preprocessing for Real Data ---

def preprocess_real_data(df: pd.DataFrame, ema_period=50):
    """Adds necessary indicators to real market data."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    resample_period = 'W'
    df_resampled = df['Close'].resample(resample_period).ohlc()

    peak_indices, _ = find_peaks(df_resampled['high'], prominence=10, distance=3)
    trough_indices, _ = find_peaks(-df_resampled['low'], prominence=10, distance=3)

    df_resampled['swing_high'] = np.nan
    df_resampled.iloc[peak_indices, df_resampled.columns.get_loc('swing_high')] = df_resampled['high'].iloc[peak_indices]

    df_resampled['swing_low'] = np.nan
    df_resampled.iloc[trough_indices, df_resampled.columns.get_loc('swing_low')] = df_resampled['low'].iloc[trough_indices]

    df = pd.merge_asof(df, df_resampled[['swing_high', 'swing_low']], left_index=True, right_index=True, direction='forward')

    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    df.ffill(inplace=True); df.dropna(inplace=True)
    return df

# --- Strategy Class ---

class FiftyPercentRetracementScalpMwFormationStrategy(Strategy):
    min_rr = 1.5
    confluence_tolerance = 20.0 # Wider tolerance

    def init(self):
        self.setup_high = None
        self.setup_low = None
        self.just_closed_short_tp = False
        self.closed_trades_count = 0
        df = self.data.df
        self.swing_highs = df.get('swing_high', df['High'])
        self.swing_lows = df.get('swing_low', df['Low'])
        self.ema = df.get('EMA', df['Close'].ewm(span=50).mean())

    def next(self):
        # --- Bag Flip Logic ---
        if len(self.closed_trades) > self.closed_trades_count:
            if self.just_closed_short_tp and self.data.Close[-1] > self.data.Open[-1]:
                self.buy()
            self.just_closed_short_tp = False
            self.closed_trades_count = len(self.closed_trades)

        if self.position: return

        # --- Simplified Stateless Entry Logic ---
        if not pd.isna(self.swing_highs.iloc[-1]): self.setup_high = self.swing_highs.iloc[-1]
        if not pd.isna(self.swing_lows.iloc[-1]): self.setup_low = self.swing_lows.iloc[-1]

        if not self.setup_high or not self.setup_low or self.setup_high <= self.setup_low:
            return

        fib_50_level = self.setup_low + (self.setup_high - self.setup_low) * 0.5

        price_touched_50 = self.data.High[-1] >= fib_50_level
        price_closed_below_50 = self.data.Close[-1] < fib_50_level

        if price_touched_50 and price_closed_below_50:
            if abs(fib_50_level - self.ema.iloc[-1]) < self.confluence_tolerance:
                sl = self.data.High[-1] * 1.02

                # TP is 50% of the rejection candle move
                rejection_candle_low = self.data.Low[-1]
                rejection_candle_high = self.data.High[-1]
                tp = rejection_candle_low + (rejection_candle_high - rejection_candle_low) * 0.5

                entry_price = self.data.Close[-1]
                if tp >= entry_price: return

                risk = abs(sl - entry_price); reward = abs(entry_price - tp)
                if risk > 0 and reward / risk >= self.min_rr:
                    self.sell(sl=sl, tp=tp)
                    self.just_closed_short_tp = True
                    self.setup_high, self.setup_low = None, None # Reset to avoid re-entry

if __name__ == '__main__':
    import os
    import json
    from backtesting import Backtest
    
    data_path = os.environ.get('BACKTEST_DATA_PATH')
    mode = os.environ.get('BACKTEST_MODE', 'standalone')
    
    if data_path and os.path.exists(data_path):
        # === STANDARDIZED MODE ===
        print(f"[Standardized Mode] Loading data from: {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.title() for c in data.columns]
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        from backtesting.lib import FractionalBacktest
        bt = FractionalBacktest(data, FiftyPercentRetracementScalpMwFormationStrategy, cash=10000, commission=.002)
        
        # In standardized mode, always run with defaults (optimization requires strategy-specific params)
        print("[Run Mode] Running single backtest with defaults...")
        stats = bt.run()
        
        # Save results
        os.makedirs('results', exist_ok=True)
        result = {
            'strategy_name': '50_percent_retracement_scalp_mw_formation',
            'return': float(stats.get('Return [%]', 0)) if not pd.isna(stats.get('Return [%]', 0)) else None,
            'sharpe': float(stats.get('Sharpe Ratio')) if stats.get('Sharpe Ratio') and not pd.isna(stats.get('Sharpe Ratio')) else None,
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)) if not pd.isna(stats.get('Max. Drawdown [%]', 0)) else None,
            'win_rate': float(stats.get('Win Rate [%]', 0)) if not pd.isna(stats.get('Win Rate [%]', 0)) else None,
            'total_trades': int(stats.get('# Trades', 0))
        }
        with open('results/temp_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Return={result['return']}%, Trades={result['total_trades']}")
    else:
        # === STANDALONE MODE (original behavior) ===
        print("[Standalone Mode] Using original data generation...")
        data = GOOG.copy()
        data_processed = preprocess_real_data(data, ema_period=50)

        bt = Backtest(data_processed, FiftyPercentRetracementScalpMwFormationStrategy, cash=100_000, commission=.002)
        stats = bt.run()

        os.makedirs('results', exist_ok=True)

        stats_dict = {
            'strategy_name': '50_percent_retracement_scalp_mw_formation',
            'return': stats.get('Return [%]'), 'sharpe': stats.get('Sharpe Ratio'),
            'max_drawdown': stats.get('Max. Drawdown [%]'), 'win_rate': stats.get('Win Rate [%]'),
            'total_trades': stats.get('# Trades')
        }

        for key, value in stats_dict.items():
            if isinstance(value, (np.integer, np.int64)): stats_dict[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)): stats_dict[key] = float(value) if not np.isnan(value) else None

        with open('results/temp_result.json', 'w') as f:
            json.dump(stats_dict, f, indent=2)

        if stats_dict.get('total_trades', 0) > 0:
            bt.plot(filename="results/50_percent_retracement_scalp_mw_formation.html", open_browser=False)
            print("Strategy executed trades and generated results/plot.")
        else:
            print("No trades were executed.")
