from backtesting import Backtest, Strategy
import pandas as pd
import pandas_ta as ta
import numpy as np
import json

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the non-parameterized data for the NewYorkCityReversalTrade strategy.
    - Converts timezone to America/New_York.
    - Calculates ADR, pre-session HOD/LOD, and market cycle proxy.
    """
    # Ensure correct column names by stripping whitespace and capitalizing
    df.columns = [col.strip().title() for col in df.columns]

    # Timezone conversion
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

    # Calculate ADR (Average Daily Range)
    daily_range = df.resample('D')['High'].max() - df.resample('D')['Low'].min()
    adr = daily_range.rolling(window=14).mean()
    # Use previous day's ADR to avoid lookahead bias
    df['ADR'] = df.index.normalize().map(adr.shift(1))

    # Calculate High/Low of the Day (HOD/LOD) before the NY session
    pre_ny_session = df[df.index.hour < 8]
    daily_hod_lod = pre_ny_session.groupby(pre_ny_session.index.date).agg(
        HOD_pre_NY=('High', 'max'),
        LOD_pre_NY=('Low', 'min')
    )
    df['HOD_pre_NY'] = df.index.normalize().map(daily_hod_lod['HOD_pre_NY'])
    df['LOD_pre_NY'] = df.index.normalize().map(daily_hod_lod['LOD_pre_NY'])

    # Add time-based features
    df['is_NY_session'] = (df.index.hour >= 8) & (df.index.hour < 11)

    # Proxy for "Level III" - sustained move
    rolling_window = 12 # 3 hours of 15m candles
    df['higher_closes'] = (df['Close'] > df['Close'].shift(1)).rolling(window=rolling_window).sum()
    df['lower_closes'] = (df['Close'] < df['Close'].shift(1)).rolling(window=rolling_window).sum()

    # Level III is defined as a high number of directional closes
    sustained_move_threshold = 9
    df['market_cycle_level_up'] = df['higher_closes'] >= sustained_move_threshold
    df['market_cycle_level_down'] = df['lower_closes'] >= sustained_move_threshold

    return df

# Wrapper functions for pandas-ta indicators to be used with self.I()
def ema(series, length):
    return ta.ema(pd.Series(series), length=length).values

def rsi(series, length):
    return ta.rsi(pd.Series(series), length=length).values

def sma(series, length):
    return ta.sma(pd.Series(series), length=length).values


class NewYorkCityReversalTrade(Strategy):
    """
    Strategy Implementation: New York City Reversal Trade
    Source: Steve Mauro Notes (strategy 6)
    """

    # Default Parameters
    ema_fast_period = 5
    ema_medium_period = 13
    ema_slow_period = 50
    ema_trend_period = 200
    rsi_period = 14
    tdi_rsi_period = 13
    tdi_fast_ma = 2
    tdi_slow_ma = 7
    adr_period = 14 # Note: Not tunable yet as it's in preprocess
    ny_session_start = 8
    ny_session_end = 11
    ema_pullaway_pct = 0.1 # Price must be at least 0.1% away from 5 EMA
    rsi_overbought = 70
    rsi_oversold = 30

    def init(self):
        # Pre-calculated, non-parameterized data
        self.adr = self.I(lambda x: x, self.data.df['ADR'])
        self.hod_pre_ny = self.I(lambda x: x, self.data.df['HOD_pre_NY'])
        self.lod_pre_ny = self.I(lambda x: x, self.data.df['LOD_pre_NY'])
        self.is_ny_session = self.I(lambda x: x, self.data.df['is_NY_session'])
        self.market_cycle_level_up = self.I(lambda x: x, self.data.df['market_cycle_level_up'])
        self.market_cycle_level_down = self.I(lambda x: x, self.data.df['market_cycle_level_down'])

        # Parameter-dependent indicators
        self.ema5 = self.I(ema, self.data.Close, self.ema_fast_period)
        self.ema13 = self.I(ema, self.data.Close, self.ema_medium_period)
        self.ema50 = self.I(ema, self.data.Close, self.ema_slow_period)
        self.ema200 = self.I(ema, self.data.Close, self.ema_trend_period)
        self.rsi = self.I(rsi, self.data.Close, self.rsi_period)

        # TDI Calculation
        tdi_rsi_series = self.I(rsi, self.data.Close, self.tdi_rsi_period)
        self.tdi_price_line = self.I(sma, tdi_rsi_series, self.tdi_fast_ma)
        self.tdi_signal_line = self.I(sma, tdi_rsi_series, self.tdi_slow_ma)

    def is_bearish_engulfing(self):
        if len(self.data.Close) < 2: return False
        return (self.data.Close[-2] > self.data.Open[-2] and # Previous is bullish
                self.data.Close[-1] < self.data.Open[-1] and # Current is bearish
                self.data.Open[-1] >= self.data.Close[-2] and
                self.data.Close[-1] < self.data.Open[-2])

    def is_bullish_engulfing(self):
        if len(self.data.Close) < 2: return False
        return (self.data.Close[-2] < self.data.Open[-2] and # Previous is bearish
                self.data.Close[-1] > self.data.Open[-1] and # Current is bullish
                self.data.Open[-1] <= self.data.Close[-2] and
                self.data.Close[-1] > self.data.Open[-2])

    def next(self):
        if self.position:
            return

        if not self.is_ny_session[-1]:
            return

        # --- Bearish Reversal Setup ---
        adr_target_hit_high = self.data.High[-1] >= (self.lod_pre_ny[-1] + self.adr[-1])
        hod_formed = self.data.High[-1] >= self.hod_pre_ny[-1]
        pullaway_from_ema = self.data.Close[-1] > self.ema5[-1] * (1 + self.ema_pullaway_pct / 100)
        is_level_3_up = self.market_cycle_level_up[-1] == True
        tdi_cross_down = self.tdi_price_line[-1] < self.tdi_signal_line[-1] and \
                         self.tdi_price_line[-2] > self.tdi_signal_line[-2]

        if (adr_target_hit_high and hod_formed and pullaway_from_ema and is_level_3_up and
            self.is_bearish_engulfing() and self.rsi[-1] > self.rsi_overbought and tdi_cross_down):

            sl = self.data.High[-1] * 1.001
            tp = self.ema50[-1]
            if tp < self.data.Close[-1]:
                self.sell(sl=sl, tp=tp)

        # --- Bullish Reversal Setup ---
        adr_target_hit_low = self.data.Low[-1] <= (self.hod_pre_ny[-1] - self.adr[-1])
        lod_formed = self.data.Low[-1] <= self.lod_pre_ny[-1]
        pullaway_from_ema_low = self.data.Close[-1] < self.ema5[-1] * (1 - self.ema_pullaway_pct / 100)
        is_level_3_down = self.market_cycle_level_down[-1] == True
        tdi_cross_up = self.tdi_price_line[-1] > self.tdi_signal_line[-1] and \
                       self.tdi_price_line[-2] < self.tdi_signal_line[-2]

        if (adr_target_hit_low and lod_formed and pullaway_from_ema_low and is_level_3_down and
            self.is_bullish_engulfing() and self.rsi[-1] < self.rsi_oversold and tdi_cross_up):

            sl = self.data.Low[-1] * 0.999
            tp = self.ema50[-1]
            if tp > self.data.Close[-1]:
                self.buy(sl=sl, tp=tp)


if __name__ == '__main__':
    import os

    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        # Create a dummy result file for CI/CD
        os.makedirs('results', exist_ok=True)
        with open('results/temp_result.json', 'w') as f:
            json.dump({'strategy_name': 'new_york_city_reversal_trade', 'error': 'Data not found'}, f)
    else:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Preprocess the data
        data = preprocess_data(data.copy()) # Use a copy to avoid modifying the original df

        # Initialize and run the backtest
        bt = Backtest(data, NewYorkCityReversalTrade, cash=100_000, commission=.002)
        stats = bt.run()

        print(stats)

        # Save results
        os.makedirs('results', exist_ok=True)

        def sanitize_stats(stats):
            """Remove non-serializable objects from stats dict."""
            sanitized = {}
            # Use a list to prevent issues with changing dict size during iteration
            for key, value in list(stats.items()):
                if isinstance(value, (pd.Series, pd.DataFrame, Strategy)):
                    continue
                elif isinstance(value, pd.Timestamp):
                    sanitized[key] = value.isoformat()
                elif isinstance(value, pd.Timedelta):
                     sanitized[key] = str(value)
                elif pd.isna(value):
                    sanitized[key] = None
                elif isinstance(value, (np.int64, np.int32)):
                    sanitized[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    sanitized[key] = float(value)
                else:
                    sanitized[key] = value
            return sanitized

        clean_stats = sanitize_stats(stats)

        with open('results/temp_result.json', 'w') as f:
            json.dump(clean_stats, f, indent=2)

        print("Backtest results saved to results/temp_result.json")

        # Generate plot
        try:
            plot_filename = 'results/new_york_city_reversal_trade.html'
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
