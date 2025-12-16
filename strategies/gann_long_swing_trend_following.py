from backtesting import Backtest, Strategy
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os

def preprocess_data(df):
    """
    Adds indicators needed for the Gann Long Swing Trend Following strategy.
    - Daily EMA for trend identification.
    - Daily ATR for volatility (accumulation/distribution).
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Calculate daily indicators
    daily_df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    daily_df['daily_ema'] = ta.ema(daily_df['Close'], length=50)
    daily_df['daily_atr'] = ta.atr(daily_df['High'], daily_df['Low'], daily_df['Close'], length=14)

    # Calculate rolling volatility percentile
    daily_df['atr_percentile'] = daily_df['daily_atr'].rolling(window=50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


    # Map daily indicators back to the 15-minute timeframe
    df = df.join(daily_df[['daily_ema', 'atr_percentile']], on=df.index.date)

    # Add rolling volume average for breakout confirmation
    df['volume_rolling_avg'] = df['Volume'].rolling(window=20).mean()

    df.ffill(inplace=True)
    df.dropna(inplace=True)

    return df

def passthrough(data):
    return data

class GannLongSwingTrendFollowing(Strategy):
    """
    Implements a trend-following strategy based on Gann's principles.
    Enters long after an accumulation period in a confirmed uptrend.
    Enters short after a distribution period in a confirmed downtrend.
    """
    stop_loss_pct = 0.03  # 3% stop loss
    profit_protect_pct = 0.03 # 3% profit to move SL to breakeven
    trailing_sl_pct = 0.05 # 5% trailing stop
    accumulation_threshold = 0.25 # ATR percentile below which is considered accumulation
    distribution_threshold = 0.75 # ATR percentile above which is considered distribution
    volume_confirmation_multiplier = 1.5 # Breakout volume must be X times the rolling average

    def init(self):
        # Pre-calculated indicators from the dataframe
        self.daily_ema = self.I(passthrough, self.data.df['daily_ema'].values)
        self.atr_percentile = self.I(passthrough, self.data.df['atr_percentile'].values)
        self.volume_rolling_avg = self.I(passthrough, self.data.df['volume_rolling_avg'].values)
        self.entry_bar_index = None

    def next(self):
        current_price = self.data.Close[-1]

        # Profit protection and trailing stop logic
        if self.position:
            pnl_pct = (current_price - self.position.avg_price) / self.position.avg_price if self.position.is_long else (self.position.avg_price - current_price) / self.position.avg_price

            # Move stop to breakeven
            if pnl_pct > self.profit_protect_pct and self.trades[0].sl == self.initial_sl:
                 self.trades[0].sl = self.position.avg_price

            # Trailing stop logic
            if self.position.is_long:
                new_sl = current_price * (1 - self.trailing_sl_pct)
                if new_sl > self.trades[0].sl:
                    self.trades[0].sl = new_sl
            elif self.position.is_short:
                new_sl = current_price * (1 + self.trailing_sl_pct)
                if new_sl < self.trades[0].sl:
                    self.trades[0].sl = new_sl

        is_uptrend = current_price > self.daily_ema[-1]
        is_downtrend = current_price < self.daily_ema[-1]
        is_accumulation = self.atr_percentile[-1] < self.accumulation_threshold
        is_distribution = self.atr_percentile[-1] > self.distribution_threshold

        # Entry logic
        if not self.position:
            is_volume_breakout = self.data.Volume[-1] > self.volume_rolling_avg[-1] * self.volume_confirmation_multiplier

            # Long entry: Uptrend, coming out of accumulation with high volume
            if is_uptrend and is_accumulation and is_volume_breakout:
                # Basic breakout: close is higher than the previous high
                if len(self.data.High) > 2 and self.data.Close[-1] > self.data.High[-2]:
                    sl = current_price * (1 - self.stop_loss_pct)
                    self.initial_sl = sl
                    self.buy(sl=sl)

            # Short entry: Downtrend, coming out of distribution with high volume
            elif is_downtrend and is_distribution and is_volume_breakout:
                # Basic breakdown: close is lower than the previous low
                if len(self.data.Low) > 2 and self.data.Close[-1] < self.data.Low[-2]:
                    sl = current_price * (1 + self.stop_loss_pct)
                    self.initial_sl = sl
                    self.sell(sl=sl)


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.title() for c in data.columns]

        # Preprocess the data
        data = preprocess_data(data)

        bt = Backtest(data, GannLongSwingTrendFollowing, cash=100000, commission=.002)

        print("Running single backtest with default parameters...")
        stats = bt.run()
        print(stats)

        # Save results to a JSON file
        os.makedirs('results', exist_ok=True)

        def sanitize_stats(stats):
            clean_stats = {}
            for key, value in stats.items():
                if isinstance(value, (pd.Series, pd.DataFrame)):
                    continue
                if pd.isna(value):
                    clean_stats[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    clean_stats[key] = value.item()
                else:
                    clean_stats[key] = value
            return clean_stats

        result_dict = sanitize_stats(stats)

        result_payload = {
            'strategy_name': 'gann_long_swing_trend_following',
            'return': result_dict.get('Return [%]'),
            'sharpe': result_dict.get('Sharpe Ratio'),
            'max_drawdown': result_dict.get('Max. Drawdown [%]'),
            'win_rate': result_dict.get('Win Rate [%]'),
            'total_trades': result_dict.get('# Trades')
        }

        with open('results/temp_result.json', 'w') as f:
            json.dump(result_payload, f, indent=4)
            f.write('\n') # Add newline at the end of the file

        print("Backtest results saved to results/temp_result.json")

        try:
            plot_filename = 'results/gann_long_swing_trend_following.html'
            bt.plot(filename=plot_filename)
            print(f"Plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")

    else:
        print(f"Data file not found at {data_path}. Please ensure it exists.")
