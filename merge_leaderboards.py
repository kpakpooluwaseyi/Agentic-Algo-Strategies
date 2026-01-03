#!/usr/bin/env python3
"""
🔄 Merge Leaderboards Script
Merges historical leaderboard backups into a unified dataset with consistent schema.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Paths
RESULTS_DIR = Path('results')
BACKUPS_DIR = RESULTS_DIR / 'backups'
LEADERBOARDS_DIR = RESULTS_DIR / 'leaderboards'
OUTPUT_FILE = RESULTS_DIR / 'leaderboard_merged.csv'
CURRENT_LEADERBOARD = RESULTS_DIR / 'leaderboard.csv'

# Target schema (unified)
TARGET_COLUMNS = [
    'timestamp',
    'strategy_name',
    'dataset_name',
    'return_pct',
    'sharpe_ratio',
    'sortino_ratio',
    'treynor_ratio',
    'beta',
    'max_drawdown_pct',
    'win_rate_pct',
    'total_trades',
    'status',
    'wfa_status',
    'wfa_oos_return_pct',
    'wfa_degradation_pct',
    'stage'  # backtest, wfa, red_team
]


def load_and_normalize(filepath: Path) -> pd.DataFrame:
    """Load a leaderboard CSV and normalize to target schema."""
    try:
        df = pd.read_csv(filepath)
        print(f"  Loaded {len(df)} rows from {filepath.name}")
        
        # Rename columns to match target schema
        rename_map = {
            'dataset': 'dataset_name',
            'return': 'return_pct',
            'sharpe': 'sharpe_ratio',
            'sortino': 'sortino_ratio',
            'treynor': 'treynor_ratio',
            'max_drawdown': 'max_drawdown_pct',
            'win_rate': 'win_rate_pct',
        }
        df = df.rename(columns=rename_map)
        
        # Add missing columns with None
        for col in TARGET_COLUMNS:
            if col not in df.columns:
                df[col] = None
        
        # Set default stage
        if 'stage' not in df.columns or df['stage'].isna().all():
            df['stage'] = 'backtest'
        
        # Select only target columns (in order)
        df = df[[c for c in TARGET_COLUMNS if c in df.columns]]
        
        return df
        
    except Exception as e:
        print(f"  ⚠️ Error loading {filepath.name}: {e}")
        return pd.DataFrame()


def merge_all_leaderboards():
    """Merge all leaderboard sources."""
    print("🔄 Merging Leaderboards...")
    
    all_dfs = []
    
    # 1. Load backups (prioritize richest data)
    if BACKUPS_DIR.exists():
        print(f"\n📁 Loading from backups/")
        for backup in sorted(BACKUPS_DIR.glob('leaderboard_*.csv'), reverse=True):
            df = load_and_normalize(backup)
            if not df.empty:
                all_dfs.append(df)
    
    # 2. Load per-dataset leaderboards
    if LEADERBOARDS_DIR.exists():
        print(f"\n📁 Loading from leaderboards/")
        for lb_file in LEADERBOARDS_DIR.glob('*.csv'):
            df = load_and_normalize(lb_file)
            if not df.empty:
                # Infer dataset from filename if not in data
                dataset_name = lb_file.stem
                if 'dataset_name' in df.columns:
                    df['dataset_name'] = df['dataset_name'].fillna(dataset_name)
                all_dfs.append(df)
    
    # 3. Load current leaderboard
    if CURRENT_LEADERBOARD.exists():
        print(f"\n📁 Loading current leaderboard")
        df = load_and_normalize(CURRENT_LEADERBOARD)
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        print("❌ No data found!")
        return
    
    # Concatenate all
    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📊 Total rows before dedup: {len(merged)}")
    
    # Deduplicate - keep latest entry per (strategy_name, dataset_name)
    merged['timestamp'] = pd.to_datetime(merged['timestamp'], errors='coerce')
    merged = merged.sort_values('timestamp', ascending=False)
    merged = merged.drop_duplicates(subset=['strategy_name', 'dataset_name'], keep='first')
    
    print(f"📊 Total rows after dedup: {len(merged)}")
    print(f"📊 Unique strategies: {merged['strategy_name'].nunique()}")
    
    # Save merged
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved to: {OUTPUT_FILE}")
    
    # Summary stats
    print("\n📈 Summary:")
    print(f"  - Total entries: {len(merged)}")
    print(f"  - Unique strategies: {merged['strategy_name'].nunique()}")
    print(f"  - SUCCESS: {len(merged[merged['status'] == 'SUCCESS'])}")
    print(f"  - ERROR: {len(merged[merged['status'] == 'ERROR'])}")
    
    # Positive returns
    positive = merged[(merged['return_pct'].notna()) & (merged['return_pct'] > 0)]
    print(f"  - Positive returns: {len(positive)}")


if __name__ == '__main__':
    merge_all_leaderboards()
