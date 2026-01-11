
import pandas as pd
from backtesting import Strategy, Backtest

# 1. Preprocessing function to add a custom column
def preprocess_for_debug(df):
    df = df.copy()
    # Sanitize columns first to avoid KeyErrors
    df.columns = [col.strip().capitalize() for col in df.columns]
    # Add a simple custom column
    df['Custom_Signal'] = df['Close'] > df['Open']
    return df

# 2. A minimal strategy to test data access
class DebugStrategy(Strategy):
    def init(self):
        print("--- Inside DebugStrategy.init() ---")
        # Check if the custom column exists in the underlying DataFrame
        if 'Custom_Signal' in self.data.df.columns:
            print("SUCCESS: 'Custom_Signal' column found in self.data.df")
            # Try to create an indicator from it
            try:
                self.signal = self.I(self.data.df['Custom_Signal'], name="CustomSignal")
                print("SUCCESS: Indicator created from custom column.")
            except Exception as e:
                print(f"ERROR: Failed to create indicator from column. Reason: {e}")
        else:
            print("ERROR: 'Custom_Signal' column NOT found in self.data.df")
            print("Available columns:", self.data.df.columns)

    def next(self):
        pass

# 3. Standalone runner
if __name__ == '__main__':
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Data file not found.")
        exit(1)

    # Process the data
    df_processed = preprocess_for_debug(df)
    df_processed.dropna(inplace=True)

    print("--- Columns before running Backtest ---")
    print(df_processed.columns)
    print(f"'Custom_Signal' in df_processed: {'Custom_Signal' in df_processed.columns}")

    # Run the backtest
    bt = Backtest(df_processed, DebugStrategy, cash=10000, commission=.002)
    bt.run()
