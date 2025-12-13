#!/usr/bin/env python3
"""
DataLoader - Standardized dataset management for backtesting
"""

import os
import json
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional


DATA_DIR = Path(__file__).parent
DATASETS_FILE = DATA_DIR / 'datasets.json'


class DataLoader:
    """Manages datasets for backtesting"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self._load_registry()
    
    def _load_registry(self):
        """Load dataset registry"""
        if DATASETS_FILE.exists():
            with open(DATASETS_FILE, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {"datasets": []}
    
    def _save_registry(self):
        """Save dataset registry"""
        with open(DATASETS_FILE, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        return [d['name'] for d in self.registry.get('datasets', [])]
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Load a dataset by name"""
        for dataset in self.registry.get('datasets', []):
            if dataset['name'] == name:
                filepath = self.data_dir / dataset['path']
                if filepath.exists():
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    return df
                else:
                    print(f"Dataset file not found: {filepath}")
                    return None
        print(f"Dataset not in registry: {name}")
        return None
    
    def download_crypto(self, symbol: str = 'BTC-USD', interval: str = '15m', 
                        period: str = '60d') -> pd.DataFrame:
        """
        Download crypto data from yfinance
        
        Args:
            symbol: Ticker symbol (e.g., 'BTC-USD', 'ETH-USD')
            interval: Data interval ('15m', '1h', '1d')
            period: Data period ('7d', '60d', '1y')
        """
        print(f"Downloading {symbol} {interval} data for {period}...")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Standardize column names
        df = df.rename(columns={
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Keep only OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Clean name for filename
        clean_symbol = symbol.replace('-', '')
        interval_name = interval.replace('m', 'min').replace('h', 'hour').replace('d', 'day')
        name = f"{clean_symbol}_{interval}"
        
        # Save to file
        subdir = 'crypto'
        filepath = f"{subdir}/{name}.csv"
        full_path = self.data_dir / filepath
        
        df.to_csv(full_path)
        print(f"Saved {len(df)} rows to {full_path}")
        
        # Add to registry
        dataset_entry = {
            'name': name,
            'symbol': symbol,
            'interval': interval,
            'path': filepath,
            'rows': len(df),
            'start': str(df.index[0]),
            'end': str(df.index[-1]),
            'downloaded': datetime.now().isoformat()
        }
        
        # Update or add entry
        existing = [d for d in self.registry['datasets'] if d['name'] == name]
        if existing:
            self.registry['datasets'] = [d for d in self.registry['datasets'] if d['name'] != name]
        self.registry['datasets'].append(dataset_entry)
        self._save_registry()
        
        return df
    
    def download_stock(self, symbol: str = 'GOOG', interval: str = '1d',
                       period: str = '2y') -> pd.DataFrame:
        """Download stock data from yfinance"""
        return self.download_crypto(symbol, interval, period)  # Same API
    
    def get_dataset_info(self, name: str) -> Optional[dict]:
        """Get metadata about a dataset"""
        for dataset in self.registry.get('datasets', []):
            if dataset['name'] == name:
                return dataset
        return None


def main():
    """Download default datasets"""
    loader = DataLoader()
    
    print("=" * 60)
    print("Downloading BTC-USD 15m data (priority)")
    print("=" * 60)
    btc_df = loader.download_crypto('BTC-USD', '15m', '60d')
    
    print(f"\n📊 BTC-USD 15m Dataset:")
    print(f"   Rows: {len(btc_df)}")
    print(f"   Date range: {btc_df.index[0]} to {btc_df.index[-1]}")
    print(f"   Price range: ${btc_df['Low'].min():.2f} - ${btc_df['High'].max():.2f}")
    
    print("\n" + "=" * 60)
    print("Available datasets:")
    print("=" * 60)
    for name in loader.list_datasets():
        info = loader.get_dataset_info(name)
        print(f"  - {name}: {info['rows']} rows ({info['start'][:10]} to {info['end'][:10]})")


if __name__ == '__main__':
    main()
