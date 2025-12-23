
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import json
import os
import numpy as np

def preprocess_data_with_daily_smas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 5, 10, and 20-day SMAs to the 15-minute dataframe.
    """
    daily_df = df['Close'].resample('D').last().to_frame()

    daily_df['SMA_5D'] = ta.sma(daily_df['Close'], length=5)
    daily_df['SMA_10D'] = ta.sma(daily_df['Close'], length=10)
    daily_df['SMA_20D'] = ta.sma(daily_df['Close'], length=20)

    df['SMA_5D'] = df.index.normalize().map(daily_df['SMA_5D'].shift(1))
    df['SMA_10D'] = df.index.normalize().map(daily_df['SMA_10D'].shift(1))
    df['SMA_20D'] = df.index.normalize().map(daily_df['SMA_20D'].shift(1))

    df.ffill(inplace=True)
    df.dropna(inplace=True)

    return df

class SmbSmaReclaimLongStrategy(Strategy):
    """
    Strategy based on reclaiming key Simple Moving Averages (SMAs) after a pullback.
    """
    lookback_period = 480
    convergence_pct_threshold = 2.0

    def init(self):
        self.sma_5d = self.data.SMA_5D
        self.sma_10d = self.data.SMA_10D
        self.sma_20d = self.data.SMA_20D

    def next(self):
        price = self.data.Close[-1]

        if self.position:
            if len(self.trades) > 0:
                trade = self.trades[0]
                new_sl = self.sma_10d[-1]
                if new_sl > trade.sl and new_sl < price:
                    trade.sl = new_sl

        if not self.position:
            was_below = False
            for i in range(1, self.lookback_period + 1):
                if len(self.data.Close) > i + 1:
                    past_price = self.data.Close[-i-1]
                    if (past_price < self.sma_5d[-i-1] and
                        past_price < self.sma_10d[-i-1] and
                        past_price < self.sma_20d[-i-1]):
                        was_below = True
                        break

            if not was_below:
                return

            sma10 = self.sma_10d[-1]
            sma20 = self.sma_20d[-1]
            if sma10 == 0 or sma20 == 0: return

            diff = abs(sma10 - sma20)
            avg = (sma10 + sma20) / 2
            convergence = (diff / avg) * 100 < self.convergence_pct_threshold

            if not convergence:
                return

            reclaimed_levels = (price > self.sma_10d[-1] and price > self.sma_20d[-1])

            if reclaimed_levels:
                stop_loss = self.sma_20d[-1]
                if price > stop_loss:
                    self.buy(sl=stop_loss, size=1)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.strip().capitalize() for c in data.columns]
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        print("Preprocessing data to add daily SMAs...")
        data = preprocess_data_with_daily_smas(data)

        data['SMA_5D'] = data['SMA_5D'].values
        data['SMA_10D'] = data['SMA_10D'].values
        data['SMA_20D'] = data['SMA_20D'].values

        bt = Backtest(data, SmbSmaReclaimLongStrategy, cash=100_000, commission=.002)

        print("Running backtest...")
        stats = bt.run()
        print(stats)

        os.makedirs('results', exist_ok=True)

        result_data = {
            'strategy_name': 'smb_sma_reclaim_long',
            'return': stats.get('Return [%]', 0),
            'sharpe': stats.get('Sharpe Ratio', 0),
            'max_drawdown': stats.get('Max. Drawdown [%]', 0),
            'win_rate': stats.get('Win Rate [%]', 0),
            'total_trades': stats.get('# Trades', 0)
        }

        for key, value in result_data.items():
            if pd.isna(value):
                result_data[key] = None

        with open('results/temp_result.json', 'w') as f:
            json.dump(result_data, f, indent=2)

        print("Backtest results saved to results/temp_result.json")

        plot_filename = 'results/smb_sma_reclaim_long.html'
        try:
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Backtest plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
