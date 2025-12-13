#!/usr/bin/env python3
"""
Run All Strategies - Executes all strategies and collects results
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

STRATEGIES_DIR = Path('strategies')
RESULTS_DIR = Path('results')
LEADERBOARD_FILE = RESULTS_DIR / 'leaderboard.csv'
TEMP_RESULT_FILE = RESULTS_DIR / 'temp_result.json'
STRATEGY_TIMEOUT = 300  # 5 minutes

def run_strategy(strategy_file):
    """Execute a single strategy"""
    strategy_name = strategy_file.stem
    print(f"\n{'='*60}")
    print(f"Running: {strategy_name}")
    print(f"{'='*60}")
    
    try:
        # Clean up temp result
        if TEMP_RESULT_FILE.exists():
            TEMP_RESULT_FILE.unlink()
        
        # Execute strategy
        result = subprocess.run(
            [sys.executable, str(strategy_file)],
            capture_output=True,
            text=True,
            timeout=STRATEGY_TIMEOUT,
            cwd=Path.cwd()
        )
        
        if result.returncode != 0:
            print(f"  ❌ ERROR: {result.stderr[:200]}")
            return {
                'strategy_name': strategy_name,
                'status': 'ERROR',
                'error': result.stderr[:200]
            }
        
        # Check for result file
        if not TEMP_RESULT_FILE.exists():
            print(f"  ⚠️ NO OUTPUT")
            return {
                'strategy_name': strategy_name,
                'status': 'NO_OUTPUT'
            }
        
        # Read results
        with open(TEMP_RESULT_FILE, 'r') as f:
            results = json.load(f)
        
        results['status'] = 'SUCCESS'
        print(f"  ✅ SUCCESS - Return: {results.get('return', 'N/A')}%, Trades: {results.get('total_trades', 'N/A')}")
        return results
        
    except subprocess.TimeoutExpired:
        print(f"  ⏰ TIMEOUT after {STRATEGY_TIMEOUT}s")
        return {'strategy_name': strategy_name, 'status': 'TIMEOUT'}
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        return {'strategy_name': strategy_name, 'status': 'EXCEPTION', 'error': str(e)}

def main():
    print("🚀 Running All Strategies")
    print("=" * 60)
    
    # Get all strategy files
    strategies = sorted(STRATEGIES_DIR.glob('*.py'))
    strategies = [s for s in strategies if not s.stem.startswith('__')]
    
    print(f"\nFound {len(strategies)} strategies to run")
    
    results_list = []
    
    for i, strategy_file in enumerate(strategies, 1):
        print(f"\n[{i}/{len(strategies)}]", end="")
        result = run_strategy(strategy_file)
        
        # Add to results
        row = {
            'timestamp': datetime.now().isoformat(),
            'strategy_name': result.get('strategy_name', 'unknown'),
            'return_pct': result.get('return', None),
            'sharpe_ratio': result.get('sharpe', None),
            'max_drawdown_pct': result.get('max_drawdown', None),
            'win_rate_pct': result.get('win_rate', None),
            'total_trades': result.get('total_trades', None),
            'status': result.get('status', 'UNKNOWN')
        }
        results_list.append(row)
        
        # Append to leaderboard
        df = pd.DataFrame([row])
        df.to_csv(LEADERBOARD_FILE, mode='a', header=False, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    df = pd.DataFrame(results_list)
    success = len(df[df['status'] == 'SUCCESS'])
    errors = len(df[df['status'] != 'SUCCESS'])
    
    print(f"Total: {len(strategies)}")
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {errors}")
    
    # Show successful strategies
    if success > 0:
        print("\n📈 Successful Strategies:")
        success_df = df[df['status'] == 'SUCCESS'][['strategy_name', 'return_pct', 'total_trades', 'win_rate_pct']]
        print(success_df.to_string(index=False))
    
    print(f"\nResults saved to: {LEADERBOARD_FILE}")

if __name__ == '__main__':
    main()
