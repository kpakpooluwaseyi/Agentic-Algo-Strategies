"""
Batch Test Runner for All 10 Strategy Rewrites
================================================
Tests all rewrites on BTC 4H data and produces summary.

Author: Antigravity (Claude Opus)
"""

import subprocess
import os
import json
from pathlib import Path

REWRITES_DIR = Path("strategies/rewrites")
DATA_PATH = "data/crypto/BTC-USDT_4h_200weeks.csv"

STRATEGIES = [
    # Original 10 from plan + bonus
    ("simple_trend_follower.py", "Simple Trend Follower"),
    ("measured_move_breakout.py", "Measured Move Breakout"),
    ("fib_retracement_strategy.py", "Fib Retracement"),
    ("liquidity_sweep_reversal.py", "Liquidity Sweep"),
    ("bb_mean_reversion.py", "BB Mean Reversion"),
    ("ma_pullback_continuation.py", "MA Pullback"),
    ("triple_reversal_pattern.py", "Triple Reversal"),
    ("range_breakout_pullback.py", "Range Breakout"),
    ("wave_correction_entry.py", "Wave Correction"),
]

def run_test(script_path, name):
    """Run a strategy test and capture results."""
    try:
        result = subprocess.run(
            ["python3", str(script_path)],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "BACKTEST_DATA_PATH": DATA_PATH}
        )
        
        # Parse output for key metrics
        output = result.stdout + result.stderr
        
        # Look for Return and Sharpe in output
        ret = sharpe = trades = 0
        for line in output.split("\n"):
            if "Return:" in line or "Return [%]:" in line:
                try:
                    ret = float(line.split(":")[-1].strip().replace("%", "").split(",")[0])
                except:
                    pass
            if "Sharpe:" in line or "Sharpe Ratio:" in line:
                try:
                    s = line.split(":")[-1].strip().split(",")[0]
                    if s != "N/A":
                        sharpe = float(s)
                except:
                    pass
            if "Trades:" in line or "# Trades:" in line:
                try:
                    trades = int(line.split(":")[-1].strip().split()[0])
                except:
                    pass
        
        return {"name": name, "return": ret, "sharpe": sharpe, "trades": trades, "status": "✅" if ret > 0 else "❌"}
    except Exception as e:
        return {"name": name, "return": 0, "sharpe": 0, "trades": 0, "status": f"ERROR: {str(e)[:30]}"}

def main():
    print("="*70)
    print("10 STRATEGY REWRITE - BATCH TEST")
    print(f"Dataset: {DATA_PATH} (BTC 4H 2021-2025)")
    print("="*70 + "\n")
    
    results = []
    
    for script, name in STRATEGIES:
        script_path = REWRITES_DIR / script
        if script_path.exists():
            print(f"Testing {name}...", end=" ")
            r = run_test(script_path, name)
            results.append(r)
            print(f"{r['status']} Return={r['return']:.1f}%, Sharpe={r['sharpe']:.2f}, Trades={r['trades']}")
        else:
            print(f"⚠️ {name}: File not found")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    profitable = [r for r in results if r['return'] > 0]
    print(f"\nProfitable: {len(profitable)}/{len(results)}")
    
    if profitable:
        print("\n🏆 PROFITABLE STRATEGIES:")
        for p in sorted(profitable, key=lambda x: -x['return']):
            print(f"   {p['name']}: Return={p['return']:.2f}%, Sharpe={p['sharpe']:.3f}")
    
    # Sort by return
    print("\n📊 ALL RESULTS (sorted by return):")
    for r in sorted(results, key=lambda x: -x['return']):
        print(f"   {r['status']} {r['name']}: Return={r['return']:.1f}%, Sharpe={r['sharpe']:.2f}, Trades={r['trades']}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/batch_rewrite_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to results/batch_rewrite_results.json")

if __name__ == "__main__":
    main()
