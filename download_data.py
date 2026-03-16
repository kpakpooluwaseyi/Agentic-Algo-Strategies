
import sys
import yfinance as yf
import pandas as pd
from pathlib import Path
import argparse

def download_data(symbol, interval, period):
    print(f"Downloading {symbol} ({interval}) for {period}...")
    try:
        # standard download without group_by
        df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
        
        if df.empty:
            print(f"Error: No data found for {symbol}")
            return
            
        print(f"Debug: Columns -> {df.columns}")

        # Handle MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            # Try to flatten by taking the first level if it matches expected columns, 
            # OR if it's (Price, Ticker), take the Price level.
            # yfinance often returns (Price, Ticker) like ('Open', 'ETH-USD')
            # We want 'Open'.
            # Inspect first element
            col0 = df.columns[0]
            if isinstance(col0, tuple):
                # Assume level 0 is the price type (Open, Close, etc)
                df.columns = [c[0] for c in df.columns]
            
        # Standardize columns
        # We expect: Open, High, Low, Close, Volume
        current_cols = set(df.columns)
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        
        # Check if we have them (case insensitive?)
        if not required.issubset(current_cols):
            print(f"Warning: Missing columns. Found: {current_cols}")
            # Try to map if possible (e.g. 'Adj Close' -> 'Close')
            # But auto_adjust=True usually gives Close as adjusted.
            
        # Select and Rename
        # intersect first
        available = list(required.intersection(current_cols))
        df = df[available]
        
        # We need all 5 for strategy? 
        # Strategy needs Open, High, Low, Close, Volume.
        # If Volume is missing (e.g. Forex), backtesting might fail or warn.
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Save to data/
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        filename = f"{symbol.replace('-', '')}_{interval}.csv"
        filepath = output_dir / filename
        
        df.to_csv(filepath)
        print(f"Saved {len(df)} rows to {filepath}")
        print(df.head())
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download market data via yfinance")
    parser.add_argument("symbol", type=str, help="Ticker symbol (e.g. BTC-USD, ETH-USD, SPY)")
    parser.add_argument("interval", type=str, default="15m", help="Timeframe (15m, 1h, 1d)")
    parser.add_argument("--period", type=str, default="60d", help="Lookback period (e.g. 60d, 1y)")
    
    args = parser.parse_args()
    download_data(args.symbol, args.interval, args.period)
