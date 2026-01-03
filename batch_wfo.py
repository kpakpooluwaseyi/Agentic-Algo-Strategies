#!/usr/bin/env python3
"""
🔧 Batch Walk-Forward Optimization Runner
Runs WFO on all strategies in the strategies folder.

Usage:
    python batch_wfo.py              # Run on all strategies
    python batch_wfo.py --dataset BTC_1h  # Specific dataset
    python batch_wfo.py --limit 10   # Only first 10
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Paths
STRATEGIES_DIR = Path(__file__).parent / 'strategies'
RESULTS_DIR = Path(__file__).parent / 'results' / 'red_team'
SUMMARY_FILE = RESULTS_DIR / 'batch_wfo_summary.json'
PYTHON_CMD = sys.executable

def get_all_strategies():
    """Get all strategy names from strategies folder."""
    strategies = []
    for py_file in STRATEGIES_DIR.glob('*.py'):
        if not py_file.stem.startswith('__'):
            strategies.append(py_file.stem)
    return sorted(strategies)

def get_completed_strategies():
    """Get strategies that already have WFO results."""
    completed = set()
    for json_file in RESULTS_DIR.glob('wfa_*_wfo.json'):
        # Extract strategy name from filename
        name = json_file.stem.replace('wfa_', '').replace('_wfo', '')
        completed.add(name)
    return completed

def run_batch_wfo(dataset: str = 'BTC_1h', limit: int = None, resume: bool = False):
    """Run WFO on all strategies using subprocess for process isolation."""
    strategies = get_all_strategies()
    
    # Resume mode: skip already completed
    skipped = 0
    if resume:
        completed = get_completed_strategies()
        original_count = len(strategies)
        strategies = [s for s in strategies if s not in completed]
        skipped = original_count - len(strategies)
        print(f"📦 Resume mode: skipping {skipped} already-completed strategies")
    
    if limit:
        strategies = strategies[:limit]
    
    print(f"🔧 Batch WFO Runner (Process Isolated)")
    print(f"{'='*60}")
    print(f"📊 Dataset: {dataset}")
    print(f"📋 Strategies to process: {len(strategies)}")
    if skipped:
        print(f"⏭️ Already completed: {skipped}")
    print(f"{'='*60}\n")
    
    if not strategies:
        print("✅ All strategies already processed!")
        return
    
    results = {
        'dataset': dataset,
        'started': datetime.now().isoformat(),
        'total': len(strategies),
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'strategies': {}
    }
    
    for i, strategy_name in enumerate(strategies, 1):
        print(f"\n[{i}/{len(strategies)}] 🔧 Processing: {strategy_name}")
        
        try:
            # Run walk_forward.py as a separate process for isolation
            cmd = [
                PYTHON_CMD, 
                "walk_forward.py", 
                "--strategy", strategy_name, 
                "--dataset", dataset, 
                "--optimize"
            ]
            
            print(f"   🚀 Running isolated process...")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check for result file
            output_file = RESULTS_DIR / f"wfa_{strategy_name}_wfo.json"
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    result = json.load(f)
                
                status = result.get('status', 'ERROR')
                if status == 'PASS':
                    results['passed'] += 1
                    print(f"   ✅ PASS")
                elif status == 'FAIL':
                    results['failed'] += 1
                    print(f"   ❌ FAIL: {result.get('fail_reasons', [])}")
                else:
                    results['errors'] += 1
                    print(f"   ⚠️ ERROR in result file")
                
                # Store summary
                results['strategies'][strategy_name] = {
                    'status': status,
                    'is_return': result.get('in_sample', {}).get('return_pct'),
                    'oos_return': result.get('out_of_sample', {}).get('return_pct'),
                    'fail_reasons': result.get('fail_reasons', [])
                }
            else:
                print(f"   ❌ Error: Process finished but no result file found.")
                print(f"   Stderr: {process.stderr[:200]}...")
                results['errors'] += 1
                results['strategies'][strategy_name] = {'status': 'MISSING_RESULT'}
            
        except Exception as e:
            print(f"   ❌ Batch runner exception: {e}")
            results['errors'] += 1
            results['strategies'][strategy_name] = {
                'status': 'EXCEPTION',
                'error': str(e)[:200]
            }
        
        # Small delay to ensure resources are released
        time.sleep(1)
    
    results['completed'] = datetime.now().isoformat()
    
    # Save summary
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"🎉 Batch WFO Complete!")
    print(f"   ✅ Passed: {results['passed']}")
    print(f"   ❌ Failed: {results['failed']}")
    print(f"   ⚠️ Errors: {results['errors']}")
    print(f"   📊 Total: {results['total']}")
    print(f"\n💾 Summary saved to: {SUMMARY_FILE}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch WFO Runner')
    parser.add_argument('--dataset', type=str, default='BTC_1h',
                        help='Dataset to use (default: BTC_1h)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of strategies to process')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-completed strategies')
    args = parser.parse_args()
    
    run_batch_wfo(dataset=args.dataset, limit=args.limit, resume=args.resume)

