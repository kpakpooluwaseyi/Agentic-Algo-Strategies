import pandas as pd
from backtesting import Strategy, Backtest
import json
import os

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the data for the Monday Morning Sentiment Reversal strategy.
    - Identifies Mondays and specific trading times.
    - Calculates the prior week's closing sentiment (strong/weak).
    """
    # Clean and correct column names, removing leading/trailing spaces and capitalizing
    df.columns = [c.strip().title() for c in df.columns]

    # Time-based features
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['hour'] = df.index.hour

    # Monday trading sessions and Tuesday
    df['is_monday_morning'] = (df['day_of_week'] == 0) & (df['hour'] >= 0) & (df['hour'] < 1) # First hour 00:00-00:59
    df['is_monday_afternoon'] = (df['day_of_week'] == 0) & (df['hour'] >= 12)
    df['is_tuesday'] = (df['day_of_week'] == 1)

    # Determine prior week's sentiment
    # W-SUN makes the week end on Sunday, which is what we want for "prior week"
    df['year_week'] = df.index.to_period('W-SUN')

    # Get the first and last closing price for each week
    weekly_agg = df.groupby('year_week')['Close'].agg(['first', 'last'])

    # Determine if the week was strong (closed higher than open) or weak
    weekly_agg['trend'] = 'strong'
    weekly_agg.loc[weekly_agg['last'] < weekly_agg['first'], 'trend'] = 'weak'

    # Shift the trend to get the *prior* week's trend for each day
    weekly_agg['prior_week_trend'] = weekly_agg['trend'].shift(1)

    # Map the prior week's trend back to the main DataFrame
    df = df.join(weekly_agg['prior_week_trend'], on='year_week')

    # Create boolean flags for the strategy
    df['prior_week_strong_close'] = (df['prior_week_trend'] == 'strong')
    df['prior_week_weak_close'] = (df['prior_week_trend'] == 'weak')

    # Clean up and drop rows that couldn't get a prior week trend
    df = df.drop(columns=['day_of_week', 'hour', 'year_week', 'prior_week_trend'])
    df = df.dropna()

    return df

class MondayMorningSentimentReversal(Strategy):
    stop_loss_pct = 3.0 # Default 3% stop loss

    def init(self):
        # Pass-through pre-calculated signals
        self.is_monday_morning = self.I(lambda x: x, self.data.df['is_monday_morning'].values, name="is_monday_morning")
        self.is_monday_afternoon = self.I(lambda x: x, self.data.df['is_monday_afternoon'].values, name="is_monday_afternoon")
        self.is_tuesday = self.I(lambda x: x, self.data.df['is_tuesday'].values, name="is_tuesday")
        self.prior_week_strong_close = self.I(lambda x: x, self.data.df['prior_week_strong_close'].values, name="prior_week_strong_close")
        self.prior_week_weak_close = self.I(lambda x: x, self.data.df['prior_week_weak_close'].values, name="prior_week_weak_close")

    def next(self):
        # === EXIT LOGIC ===
        # Exit on Monday afternoon or Tuesday if a position is open
        if self.position:
            if self.is_monday_afternoon[-1] or self.is_tuesday[-1]:
                self.position.close()
                return

        # === ENTRY LOGIC ===
        # No new entries if a position is already open
        if self.position:
            return

        # Check if it's Monday morning for a potential entry
        if self.is_monday_morning[-1]:
            sl_price = 0

            # SHORT ENTRY: If prior week was strong, sell the Monday morning rally
            if self.prior_week_strong_close[-1]:
                # Simple check for a "rally": current close is higher than open
                if self.data.Close[-1] > self.data.Open[-1]:
                    sl_price = self.data.Close[-1] * (1 + self.stop_loss_pct / 100)
                    self.sell(sl=sl_price)

            # LONG ENTRY: If prior week was weak, buy the Monday morning dip
            elif self.prior_week_weak_close[-1]:
                 # Simple check for a "dip": current close is lower than open
                if self.data.Close[-1] < self.data.Open[-1]:
                    sl_price = self.data.Close[-1] * (1 - self.stop_loss_pct / 100)
                    if sl_price > 0: # Ensure stop loss is a valid price
                        self.buy(sl=sl_price)


if __name__ == '__main__':
    data_path = 'data/crypto/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    processed_data = preprocess_data(data.copy())

    os.makedirs('results', exist_ok=True)

    if processed_data.empty:
        print("Warning: Processed data is empty after preprocessing. No backtest will be run.")
        # Create a default result file
        result = {
            'strategy_name': 'monday_morning_sentiment_reversal',
            'return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'error': 'Processed data was empty, likely due to a short dataset.'
        }
    else:
        bt = Backtest(processed_data, MondayMorningSentimentReversal, cash=100000, commission=.002)
        stats = bt.run()
        print(stats)
        result = {
            'strategy_name': 'monday_morning_sentiment_reversal',
            'return': float(stats.get('Return [%]', 0)) if not pd.isna(stats.get('Return [%]', 0)) else None,
            'sharpe': float(stats.get('Sharpe Ratio')) if stats.get('Sharpe Ratio') and not pd.isna(stats.get('Sharpe Ratio')) else None,
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)) if not pd.isna(stats.get('Max. Drawdown [%]', 0)) else None,
            'win_rate': float(stats.get('Win Rate [%]', 0)) if not pd.isna(stats.get('Win Rate [%]', 0)) else None,
            'total_trades': int(stats.get('# Trades', 0))
        }
        try:
            bt.plot(filename='results/monday_morning_sentiment_reversal.html', open_browser=False)
            print("Plot saved to results/monday_morning_sentiment_reversal.html")
        except Exception as e:
            print(f"Could not generate plot: {e}")

    with open('results/temp_result.json', 'w') as f:
        json.dump(result, f, indent=2)
        f.write('\n')

    print("Backtest results saved to results/temp_result.json")
