#!/usr/bin/env python3
"""
Run Standardized Backtest - Test all strategies on the SAME dataset
Uses subprocess with BACKTEST_DATA_PATH environment variable

Usage: 
    python run_standardized_backtest.py --dataset BTCUSD_15m
    python run_standardized_backtest.py --dataset BTC_4h --verbose
    python run_standardized_backtest.py --dataset BTC_1h --timeout 120 --stream
"""

import os
import sys
import json
import argparse
import subprocess
import time
import threading
import signal
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data.loader import DataLoader

STRATEGIES_DIR = Path('strategies')
RESULTS_DIR = Path('results')
LEADERBOARDS_DIR = RESULTS_DIR / 'leaderboards'
TEMP_RESULT_FILE = RESULTS_DIR / 'temp_result.json'
LOG_FILE = RESULTS_DIR / 'backtest_progress.log'
STRATEGY_TIMEOUT = 300

# Global flags
VERBOSE = False
STREAM_OUTPUT = False
OPTIMIZE_MODE = False


def log(message: str, level: str = "INFO"):
    """Log with timestamp to both console and file"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    formatted = f"[{timestamp}] [{level}] {message}"
    
    if VERBOSE or level in ["ERROR", "WARNING"]:
        print(formatted)
    
    # Always write to log file
    with open(LOG_FILE, 'a') as f:
        f.write(formatted + "\n")


def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024 / 1024  # Convert to MB on macOS
    except:
        return 0


def stream_output(process, prefix: str):
    """Stream subprocess output in real-time"""
    def stream_pipe(pipe, label):
        for line in iter(pipe.readline, ''):
            if line:
                log(f"{prefix} [{label}]: {line.strip()}", "STREAM")
    
    stdout_thread = threading.Thread(target=stream_pipe, args=(process.stdout, 'out'))
    stderr_thread = threading.Thread(target=stream_pipe, args=(process.stderr, 'err'))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    return stdout_thread, stderr_thread


# =============================================================================
# IMPORT-BASED RUNNER (New approach - no strategy modifications needed)
# =============================================================================

import ast
import importlib
import numpy as np
from backtesting import Backtest

def discover_strategy_classes() -> dict:
    """Auto-discover Strategy class names from all strategy files using AST"""
    registry = {}
    
    for filepath in sorted(STRATEGIES_DIR.glob('*.py')):
        if filepath.stem.startswith('__'):
            continue
        
        try:
            content = filepath.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == 'Strategy':
                            registry[filepath.stem] = node.name
                            break
        except Exception as e:
            log(f"Could not parse {filepath.stem}: {e}", "WARNING")
    
    return registry


def run_strategy_import(strategy_name: str, class_name: str, data: pd.DataFrame, 
                       optimize: bool = False) -> dict:
    """Run a strategy using direct import instead of subprocess"""
    start_time = time.time()
    
    mode_str = "OPTIMIZE" if optimize else "RUN"
    print(f"\n{'='*60}")
    print(f"Running: {strategy_name} [{mode_str}]")
    print(f"Rows: {len(data)}")
    print(f"{'='*60}")
    
    log(f"START: {strategy_name} [import mode]")
    log(f"  Memory: {get_memory_usage():.1f}MB")
    
    try:
        # Import the strategy module
        log(f"  Importing strategies.{strategy_name}...")
        module = importlib.import_module(f"strategies.{strategy_name}")
        
        # Get the Strategy class
        StrategyClass = getattr(module, class_name)
        log(f"  Found class: {class_name}")
        
        # Create backtest
        bt = Backtest(data, StrategyClass, cash=10000, commission=.002)
        
        # Run or optimize
        if optimize:
            log(f"  Running optimization...")
            # Note: For optimization, we'd need strategy-specific params
            # For now, just run with defaults
            stats = bt.run()
        else:
            log(f"  Running single backtest...")
            stats = bt.run()
        
        duration = time.time() - start_time
        
        # Extract results
        def sanitize_value(v):
            if isinstance(v, (np.integer, np.int64)):
                return int(v)
            elif isinstance(v, (np.floating, np.float64)):
                return None if np.isnan(v) else float(v)
            elif isinstance(v, (pd.Series, pd.DataFrame)):
                return None
            elif v is None:
                return None
            return v
        
        ret = sanitize_value(stats.get('Return [%]', 0))
        trades = sanitize_value(stats.get('# Trades', 0))
        
        print(f"  ✅ SUCCESS - Return: {ret:.2f}%, Trades: {trades} ({duration:.1f}s)")
        log(f"  SUCCESS: Return={ret}, Trades={trades}, Duration={duration:.1f}s")
        
        return {
            'strategy_name': strategy_name,
            'return': ret,
            'sharpe': sanitize_value(stats.get('Sharpe Ratio')),
            'max_drawdown': sanitize_value(stats.get('Max. Drawdown [%]')),
            'win_rate': sanitize_value(stats.get('Win Rate [%]')),
            'total_trades': trades,
            'status': 'SUCCESS',
            'duration': duration
        }
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)[:200]
        print(f"  ❌ ERROR: {error_msg[:100]}")
        log(f"  ERROR: {error_msg}", "ERROR")
        return {
            'strategy_name': strategy_name,
            'status': 'ERROR',
            'error': error_msg,
            'duration': duration
        }


def run_strategy_subprocess(strategy_file: Path, data_path: str, dataset_name: str, timeout: int = STRATEGY_TIMEOUT, optimize: bool = False) -> dict:
    """Run a single strategy as subprocess with data path in env"""
    strategy_name = strategy_file.stem
    start_time = time.time()
    
    mode_str = "OPTIMIZE" if optimize else "RUN"
    print(f"\n{'='*60}")
    print(f"Running: {strategy_name} [{mode_str}]")
    print(f"Dataset: {dataset_name}")
    print(f"Timeout: {timeout}s")
    print(f"{'='*60}")
    
    log(f"START: {strategy_name} on {dataset_name}")
    log(f"  Memory: {get_memory_usage():.1f}MB")
    
    try:
        # Step 1: Clean up temp result
        step_start = time.time()
        log(f"  Step 1: Cleaning temp files...")
        if TEMP_RESULT_FILE.exists():
            TEMP_RESULT_FILE.unlink()
        log(f"  Step 1: Done ({time.time() - step_start:.2f}s)")
        
        # Step 2: Set environment variables
        step_start = time.time()
        log(f"  Step 2: Setting environment...")
        env = os.environ.copy()
        env['BACKTEST_DATA_PATH'] = str(data_path)
        env['BACKTEST_DATASET_NAME'] = dataset_name
        env['BACKTEST_MODE'] = 'optimize' if optimize else 'run'
        log(f"  Step 2: Done ({time.time() - step_start:.2f}s)")
        
        # Step 3: Execute strategy
        step_start = time.time()
        log(f"  Step 3: Executing strategy subprocess...")
        log(f"    Command: {sys.executable} {strategy_file}")
        log(f"    Working dir: {Path.cwd()}")
        
        if STREAM_OUTPUT:
            # Use Popen for real-time streaming
            process = subprocess.Popen(
                [sys.executable, str(strategy_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path.cwd(),
                env=env
            )
            stream_output(process, strategy_name[:20])
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                raise
            result = type('Result', (), {'returncode': returncode, 'stdout': stdout, 'stderr': stderr})()
        else:
            result = subprocess.run(
                [sys.executable, str(strategy_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd(),
                env=env
            )
        
        exec_time = time.time() - step_start
        log(f"  Step 3: Subprocess complete ({exec_time:.2f}s)")
        log(f"    Return code: {result.returncode}")
        
        if result.stdout and VERBOSE:
            log(f"    Stdout: {result.stdout[:200]}...")
        if result.stderr:
            log(f"    Stderr: {result.stderr[:200]}...", "WARNING")
        
        # Step 4: Check results
        step_start = time.time()
        log(f"  Step 4: Checking results...")
        
        if result.returncode != 0:
            if not TEMP_RESULT_FILE.exists():
                error_msg = result.stderr[:200] if result.stderr else "Unknown error"
                print(f"  ❌ ERROR: {error_msg[:100]}")
                log(f"  FAILED: {error_msg[:100]}", "ERROR")
                return {
                    'strategy_name': strategy_name,
                    'dataset': dataset_name,
                    'status': 'ERROR',
                    'error': error_msg,
                    'duration': time.time() - start_time
                }
        
        if not TEMP_RESULT_FILE.exists():
            print(f"  ⚠️ NO OUTPUT")
            log(f"  NO OUTPUT: No temp_result.json found", "WARNING")
            return {
                'strategy_name': strategy_name,
                'dataset': dataset_name,
                'status': 'NO_OUTPUT',
                'duration': time.time() - start_time
            }
        
        # Step 5: Parse results
        log(f"  Step 5: Parsing results...")
        with open(TEMP_RESULT_FILE, 'r') as f:
            results = json.load(f)
        
        ret = results.get('return', 0)
        trades = results.get('total_trades', 0)
        duration = time.time() - start_time
        
        print(f"  ✅ SUCCESS - Return: {ret:.2f}%, Trades: {trades} ({duration:.1f}s)")
        log(f"  SUCCESS: Return={ret:.2f}%, Trades={trades}, Duration={duration:.1f}s")
        
        return {
            'strategy_name': strategy_name,
            'dataset': dataset_name,
            'return': results.get('return', 0),
            'sharpe': results.get('sharpe', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'win_rate': results.get('win_rate', 0),
            'total_trades': results.get('total_trades', 0),
            'status': 'SUCCESS',
            'duration': duration
        }
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"  ⏰ TIMEOUT after {timeout}s")
        log(f"  TIMEOUT: {strategy_name} exceeded {timeout}s", "ERROR")
        return {'strategy_name': strategy_name, 'dataset': dataset_name, 'status': 'TIMEOUT', 'duration': duration}
    except Exception as e:
        duration = time.time() - start_time
        print(f"  ❌ EXCEPTION: {str(e)[:100]}")
        log(f"  EXCEPTION: {str(e)}", "ERROR")
        return {'strategy_name': strategy_name, 'dataset': dataset_name, 'status': 'EXCEPTION', 'error': str(e), 'duration': duration}


def main():
    global VERBOSE, STREAM_OUTPUT, STRATEGY_TIMEOUT
    
    parser = argparse.ArgumentParser(description='Run standardized backtests with progress logging')
    parser.add_argument('--dataset', type=str, default='BTCUSD_15m',
                        help='Dataset name (e.g., BTCUSD_15m, BTC_4h)')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available datasets')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging to console')
    parser.add_argument('--stream', action='store_true',
                        help='Stream subprocess output in real-time')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout per strategy in seconds (default: 300)')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Run only a specific strategy (by name or partial match)')
    parser.add_argument('--optimize', action='store_true',
                        help='Run optimization instead of single run (slower, finds best params)')
    args = parser.parse_args()
    
    # Set global flags
    VERBOSE = args.verbose
    STREAM_OUTPUT = args.stream
    STRATEGY_TIMEOUT = args.timeout
    OPTIMIZE_MODE = args.optimize
    
    # Initialize log file
    with open(LOG_FILE, 'w') as f:
        f.write(f"=== Backtest Log Started: {datetime.now().isoformat()} ===\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Verbose: {VERBOSE}, Stream: {STREAM_OUTPUT}, Timeout: {STRATEGY_TIMEOUT}s\n\n")
    
    loader = DataLoader()
    
    if args.list_datasets:
        print("Available datasets:")
        for name in loader.list_datasets():
            info = loader.get_dataset_info(name)
            print(f"  - {name}: {info['rows']} rows")
        return
    
    # Get dataset info
    print(f"🔧 Standardized Backtest Runner")
    print(f"=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Timeout: {args.timeout}s per strategy")
    print(f"Verbose: {args.verbose}, Stream: {args.stream}")
    print(f"Log file: {LOG_FILE}")
    
    info = loader.get_dataset_info(args.dataset)
    if info is None:
        print(f"❌ Dataset '{args.dataset}' not found!")
        print(f"Available: {loader.list_datasets()}")
        return
    
    data_path = Path(__file__).parent / 'data' / info['path']
    print(f"Data file: {data_path}")
    print(f"Rows: {info['rows']}")
    print(f"Date range: {info['start'][:10]} to {info['end'][:10]}")
    
    # Get all strategies
    strategies = sorted(STRATEGIES_DIR.glob('*.py'))
    strategies = [s for s in strategies if not s.stem.startswith('__')]
    
    # Filter by strategy name if specified
    if args.strategy:
        strategies = [s for s in strategies if args.strategy.lower() in s.stem.lower()]
        if not strategies:
            print(f"❌ No strategies matching '{args.strategy}'")
            return
    
    # Discover Strategy class names
    print(f"\n🔍 Discovering Strategy classes...")
    registry = discover_strategy_classes()
    print(f"   Found {len(registry)} Strategy classes")
    
    # Filter to only strategies we found classes for
    strategies = [s for s in strategies if s.stem in registry]
    
    print(f"\n📊 Running {len(strategies)} strategies")
    print(f"💡 Monitor progress: tail -f {LOG_FILE}\n")
    
    # Load data once (shared across all strategies)
    print(f"📂 Loading data from: {data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.title() for c in data.columns]
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    print(f"   Loaded {len(data)} rows\n")
    
    # Create leaderboard directory
    leaderboard_dir = LEADERBOARDS_DIR / args.dataset
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_file = leaderboard_dir / 'leaderboard.csv'
    
    # Run all strategies
    results_list = []
    total_start = time.time()
    
    for i, strategy_file in enumerate(strategies, 1):
        strategy_name = strategy_file.stem
        class_name = registry[strategy_name]
        
        print(f"\n[{i}/{len(strategies)}]", end="")
        result = run_strategy_import(strategy_name, class_name, data.copy(), optimize=args.optimize)
        
        row = {
            'timestamp': datetime.now().isoformat(),
            'strategy_name': result.get('strategy_name', 'unknown'),
            'dataset': args.dataset,
            'return_pct': result.get('return', None),
            'sharpe_ratio': result.get('sharpe', None),
            'max_drawdown_pct': result.get('max_drawdown', None),
            'win_rate_pct': result.get('win_rate', None),
            'total_trades': result.get('total_trades', None),
            'status': result.get('status', 'UNKNOWN'),
            'duration_s': result.get('duration', None)
        }
        results_list.append(row)
    
    # Save leaderboard
    df = pd.DataFrame(results_list)
    df.to_csv(leaderboard_file, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY - {args.dataset}")
    print("=" * 60)
    
    success_df = df[df['status'] == 'SUCCESS']
    print(f"Total: {len(strategies)}")
    print(f"✅ Success: {len(success_df)}")
    print(f"❌ Failed: {len(df) - len(success_df)}")
    
    if len(success_df) > 0:
        # Sort by return
        success_df = success_df.sort_values('return_pct', ascending=False)
        
        print(f"\n🏆 Top 5 Strategies on {args.dataset}:")
        for i, (_, row) in enumerate(success_df.head(5).iterrows(), 1):
            trades = int(row['total_trades']) if pd.notna(row['total_trades']) else 0
            ret = row['return_pct'] if pd.notna(row['return_pct']) else 0
            print(f"  {i}. {row['strategy_name']}: {ret:+.2f}% ({trades} trades)")
        
        print(f"\n📉 Bottom 5 Strategies on {args.dataset}:")
        for i, (_, row) in enumerate(success_df.tail(5).iterrows(), 1):
            trades = int(row['total_trades']) if pd.notna(row['total_trades']) else 0
            ret = row['return_pct'] if pd.notna(row['return_pct']) else 0
            print(f"  {i}. {row['strategy_name']}: {ret:+.2f}% ({trades} trades)")
    
    print(f"\nResults saved to: {leaderboard_file}")


if __name__ == '__main__':
    main()
