#!/usr/bin/env python3
"""
Full Strategy Audit Script
Tests ALL strategies across multiple datasets with $1M cash.
"""

import sys
import os
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import warnings
warnings.filterwarnings('ignore')

# Configuration
CASH = 1_000_000
COMMISSION = 0.002
TIMEOUT_PER_STRATEGY = 60  # seconds
MAX_WORKERS = 4

DATASETS = [
    "data/crypto/BTC-USDT_1h_200weeks.csv",
    "data/crypto/BTC-USDT_15m_160weeks.csv",
    "data/crypto/ETHUSD_15m.csv",
    "data/crypto/SOLUSD_15m.csv",
    "data/crypto/XRPUSD_15m.csv",
    "data/equities/SPY_15m.csv",
    "data/equities/QQQ_15m.csv",
    "data/forex/EURUSD=X_15m.csv",
    "data/commodities/GLD_15m.csv",
]

def load_data(path):
    """Load and prepare OHLC data."""
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.columns = [c.strip().capitalize() for c in df.columns]
    # Ensure required columns
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            return None
    return df

def discover_strategies(strategies_dir):
    """Find all strategy files and their classes."""
    import importlib.util
    from backtesting import Strategy
    
    strategies = []
    strategies_path = Path(strategies_dir)
    
    for py_file in sorted(strategies_path.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "base.py":
            continue
        
        module_name = py_file.stem
        try:
            # Load module directly from file path
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find Strategy subclasses
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    issubclass(obj, Strategy) and 
                    obj is not Strategy and
                    not name.startswith("_")):
                    strategies.append({
                        "file": str(py_file),
                        "module": module_name,
                        "class_name": name,
                        "class": obj
                    })
        except Exception as e:
            # Silently skip problematic files
            pass
    
    return strategies

def run_single_backtest(strategy_info, data, dataset_name):
    """Run a single backtest and return results."""
    from backtesting.lib import FractionalBacktest
    
    try:
        bt = FractionalBacktest(data, strategy_info["class"], 
                               cash=CASH, commission=COMMISSION)
        stats = bt.run()
        
        return {
            "status": "success",
            "return_pct": float(stats["Return [%]"]) if pd.notna(stats["Return [%]"]) else 0,
            "sharpe": float(stats["Sharpe Ratio"]) if pd.notna(stats["Sharpe Ratio"]) else 0,
            "max_dd": float(stats["Max. Drawdown [%]"]) if pd.notna(stats["Max. Drawdown [%]"]) else 0,
            "trades": int(stats["# Trades"]) if pd.notna(stats["# Trades"]) else 0,
            "win_rate": float(stats["Win Rate [%]"]) if pd.notna(stats["Win Rate [%]"]) else 0,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)[:100]
        }

def main():
    print("="*80)
    print("FULL STRATEGY AUDIT")
    print(f"Started: {datetime.now()}")
    print(f"Cash: ${CASH:,}")
    print(f"Datasets: {len(DATASETS)}")
    print("="*80)
    
    # Load all datasets first
    print("\n📊 Loading datasets...")
    datasets = {}
    for ds_path in DATASETS:
        if os.path.exists(ds_path):
            data = load_data(ds_path)
            if data is not None:
                ds_name = Path(ds_path).stem
                datasets[ds_name] = data
                print(f"  ✅ {ds_name}: {len(data):,} rows")
            else:
                print(f"  ❌ {ds_path}: Invalid format")
        else:
            print(f"  ❌ {ds_path}: Not found")
    
    # Discover strategies
    print("\n🔍 Discovering strategies...")
    strategies = discover_strategies("strategies")
    print(f"  Found {len(strategies)} strategy classes")
    
    # Run audit
    print(f"\n🚀 Running audit ({len(strategies)} strategies x {len(datasets)} datasets)...")
    print(f"   Estimated backtests: {len(strategies) * len(datasets)}")
    
    results = []
    total = len(strategies) * len(datasets)
    completed = 0
    start_time = time.time()
    
    for strat in strategies:
        strat_results = {
            "strategy": strat["class_name"],
            "module": strat["module"],
            "datasets": {}
        }
        
        for ds_name, data in datasets.items():
            result = run_single_backtest(strat, data, ds_name)
            strat_results["datasets"][ds_name] = result
            completed += 1
            
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (total - completed) / rate
                print(f"   Progress: {completed}/{total} ({completed/total*100:.1f}%) - ETA: {remaining/60:.1f} min")
        
        # Calculate aggregate metrics
        successful = [r for r in strat_results["datasets"].values() if r["status"] == "success"]
        if successful:
            strat_results["avg_return"] = sum(r["return_pct"] for r in successful) / len(successful)
            strat_results["avg_sharpe"] = sum(r["sharpe"] for r in successful) / len(successful)
            strat_results["avg_trades"] = sum(r["trades"] for r in successful) / len(successful)
            strat_results["success_rate"] = len(successful) / len(datasets)
        else:
            strat_results["avg_return"] = 0
            strat_results["avg_sharpe"] = 0
            strat_results["avg_trades"] = 0
            strat_results["success_rate"] = 0
        
        results.append(strat_results)
    
    # Save results
    print("\n📁 Saving results...")
    
    # Full JSON
    with open("results/full_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create leaderboard
    leaderboard = sorted(results, key=lambda x: x["avg_sharpe"], reverse=True)
    
    # Save CSV leaderboard
    lb_data = []
    for r in leaderboard:
        lb_data.append({
            "strategy": r["strategy"],
            "module": r["module"],
            "avg_sharpe": round(r["avg_sharpe"], 4),
            "avg_return": round(r["avg_return"], 2),
            "avg_trades": round(r["avg_trades"], 1),
            "success_rate": round(r["success_rate"], 2)
        })
    
    pd.DataFrame(lb_data).to_csv("results/strategy_leaderboard.csv", index=False)
    
    # Print top 20
    print("\n" + "="*80)
    print("TOP 20 STRATEGIES BY SHARPE RATIO")
    print("="*80)
    print(f"{'Rank':<5} {'Strategy':<45} {'Sharpe':<10} {'Return%':<10} {'Trades':<8}")
    print("-"*80)
    for i, r in enumerate(leaderboard[:20], 1):
        print(f"{i:<5} {r['strategy'][:44]:<45} {r['avg_sharpe']:<10.3f} {r['avg_return']:<10.2f} {r['avg_trades']:<8.0f}")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)
    print(f"Duration: {elapsed/60:.1f} minutes")
    print(f"Strategies tested: {len(strategies)}")
    print(f"Total backtests: {completed}")
    print(f"Results saved to: results/full_audit_results.json")
    print(f"Leaderboard saved to: results/strategy_leaderboard.csv")

if __name__ == "__main__":
    main()
