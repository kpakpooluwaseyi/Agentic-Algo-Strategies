#!/usr/bin/env python3
"""
🔬 Walk-Forward Analysis Module
================================
Validates trading strategies by testing on out-of-sample data to detect overfitting.

Two validation modes:
1. Single Split (70/30): Fast initial screening
2. Rolling Window: More robust validation for graduated strategies

Usage:
    python walk_forward.py --strategy silver_bullet_fvg_retest --dataset BTC_15m
    python walk_forward.py --strategy silver_bullet_fvg_retest --rolling --windows 3
"""

import ast
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from backtesting import Backtest

# Paths
STRATEGIES_DIR = Path(__file__).parent / 'strategies'
DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'results' / 'red_team'

sys.path.insert(0, str(Path(__file__).parent))


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis for strategy validation.
    
    Detects overfitting by comparing in-sample vs out-of-sample performance.
    """
    
    def __init__(
        self,
        in_sample_ratio: float = 0.70,
        min_oos_trades: int = 3,
        overfit_threshold: float = 0.5,
        rolling_windows: int = 3,
        rolling_step_ratio: float = 0.15
    ):
        """
        Args:
            in_sample_ratio: Fraction of data for training (default 0.70 = 70%)
            min_oos_trades: Minimum trades required in OOS period
            overfit_threshold: Max acceptable degradation (0.5 = 50% loss OK)
            rolling_windows: Number of windows for rolling WFA
            rolling_step_ratio: How much to slide each window (as fraction of data)
        """
        self.in_sample_ratio = in_sample_ratio
        self.out_of_sample_ratio = 1 - in_sample_ratio
        self.min_oos_trades = min_oos_trades
        self.overfit_threshold = overfit_threshold
        self.rolling_windows = rolling_windows
        self.rolling_step_ratio = rolling_step_ratio
        
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def discover_strategy_class(self, strategy_name: str) -> Tuple[Optional[type], Optional[callable], Optional[str]]:
        """
        Discover Strategy class and preprocess_data function from strategy file.
        
        Returns: (strategy_class, preprocess_func, error_message)
        """
        filepath = STRATEGIES_DIR / f"{strategy_name}.py"
        
        if not filepath.exists():
            return None, None, f"Strategy file not found: {filepath}"
        
        try:
            content = filepath.read_text()
            tree = ast.parse(content)
            
            class_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == 'Strategy':
                            class_name = node.name
                            break
            
            if not class_name:
                return None, None, f"No Strategy class found in {strategy_name}"
            
            # Import the module and get the class
            module = importlib.import_module(f"strategies.{strategy_name}")
            strategy_class = getattr(module, class_name)
            
            # Check if module has preprocess_data function
            preprocess_func = getattr(module, 'preprocess_data', None)
            
            return strategy_class, preprocess_func, None
            
        except Exception as e:
            return None, None, f"Error loading {strategy_name}: {e}"
    
    def load_data(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load dataset by name."""
        datasets_file = DATA_DIR / 'datasets.json'
        
        if not datasets_file.exists():
            print(f"❌ datasets.json not found")
            return None
        
        with open(datasets_file, 'r') as f:
            registry = json.load(f)
        
        for dataset in registry.get('datasets', []):
            if dataset['name'] == dataset_name:
                filepath = DATA_DIR / dataset['path']
                if filepath.exists():
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    # Capitalize column names for backtesting.py compatibility
                    df.columns = [c.title() for c in df.columns]
                    # Ensure index is DatetimeIndex
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    return df
                else:
                    print(f"❌ Dataset file not found: {filepath}")
                    return None
        
        print(f"❌ Dataset not in registry: {dataset_name}")
        return None
    
    def _run_backtest(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        label: str = "",
        preprocess_func: callable = None
    ) -> Dict:
        """Run a single backtest and return results."""
        try:
            # Apply preprocessing if function exists
            if preprocess_func is not None:
                try:
                    data = preprocess_func(data.copy())
                except Exception as e:
                    return {
                        'label': label,
                        'status': 'ERROR',
                        'error': f'Preprocessing failed: {str(e)[:150]}'
                    }
            
            bt = Backtest(data, strategy_class, cash=100000, commission=0.002)
            stats = bt.run()
            
            def sanitize(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return None
                if isinstance(v, (np.integer, np.int64)):
                    return int(v)
                if isinstance(v, (np.floating, np.float64)):
                    return float(v)
                return v
            
            return {
                'label': label,
                'return_pct': sanitize(stats.get('Return [%]', 0)),
                'sharpe': sanitize(stats.get('Sharpe Ratio')),
                'max_drawdown_pct': sanitize(stats.get('Max. Drawdown [%]')),
                'win_rate_pct': sanitize(stats.get('Win Rate [%]')),
                'trades': sanitize(stats.get('# Trades', 0)),
                'start_date': str(data.index[0]),
                'end_date': str(data.index[-1]),
                'rows': len(data),
                'status': 'SUCCESS'
            }
        except Exception as e:
            return {
                'label': label,
                'status': 'ERROR',
                'error': str(e)[:200]
            }
    
    def run_single_split(
        self,
        strategy_name: str,
        data: pd.DataFrame
    ) -> Dict:
        """
        Run single 70/30 Walk-Forward Analysis.
        
        Returns dict with in_sample, out_of_sample results and pass/fail status.
        """
        strategy_class, preprocess_func, error = self.discover_strategy_class(strategy_name)
        if error:
            return {'strategy': strategy_name, 'status': 'ERROR', 'error': error}
        
        if preprocess_func:
            print(f"📦 Found preprocessing function for {strategy_name}")
        
        # Split data temporally
        split_idx = int(len(data) * self.in_sample_ratio)
        in_sample_data = data.iloc[:split_idx].copy()
        out_of_sample_data = data.iloc[split_idx:].copy()
        
        print(f"\n{'='*60}")
        print(f"🔬 Walk-Forward Analysis: {strategy_name}")
        print(f"{'='*60}")
        print(f"📊 Total rows: {len(data)}")
        print(f"📈 In-sample: {len(in_sample_data)} rows ({self.in_sample_ratio*100:.0f}%)")
        print(f"   Date range: {in_sample_data.index[0]} to {in_sample_data.index[-1]}")
        print(f"📉 Out-of-sample: {len(out_of_sample_data)} rows ({self.out_of_sample_ratio*100:.0f}%)")
        print(f"   Date range: {out_of_sample_data.index[0]} to {out_of_sample_data.index[-1]}")
        
        # Run in-sample backtest
        print(f"\n⏳ Running in-sample backtest...")
        is_result = self._run_backtest(strategy_class, in_sample_data, "in_sample", preprocess_func)
        
        if is_result['status'] == 'ERROR':
            return {
                'strategy': strategy_name,
                'status': 'ERROR',
                'error': is_result.get('error', 'Unknown error in in-sample'),
                'in_sample': is_result
            }
        
        print(f"   Return: {is_result['return_pct']:.2f}%, Trades: {is_result['trades']}")
        
        # Run out-of-sample backtest
        print(f"⏳ Running out-of-sample backtest...")
        oos_result = self._run_backtest(strategy_class, out_of_sample_data, "out_of_sample", preprocess_func)
        
        if oos_result['status'] == 'ERROR':
            return {
                'strategy': strategy_name,
                'status': 'ERROR',
                'error': oos_result.get('error', 'Unknown error in out-of-sample'),
                'in_sample': is_result,
                'out_of_sample': oos_result
            }
        
        print(f"   Return: {oos_result['return_pct']:.2f}%, Trades: {oos_result['trades']}")
        
        # Calculate degradation
        is_return = is_result['return_pct'] or 0
        oos_return = oos_result['return_pct'] or 0
        oos_trades = oos_result['trades'] or 0
        
        # Determine pass/fail
        passed = True
        fail_reasons = []
        
        # Check 1: OOS must have positive return
        if oos_return <= 0:
            passed = False
            fail_reasons.append(f"OOS return negative: {oos_return:.2f}%")
        
        # Check 2: Minimum trades in OOS
        if oos_trades < self.min_oos_trades:
            passed = False
            fail_reasons.append(f"Too few OOS trades: {oos_trades} < {self.min_oos_trades}")
        
        # Check 3: Degradation threshold (only if IS was profitable)
        if is_return > 0 and oos_return > 0:
            degradation = 1 - (oos_return / is_return)
            if degradation > self.overfit_threshold:
                passed = False
                fail_reasons.append(f"High degradation: {degradation*100:.1f}% > {self.overfit_threshold*100:.0f}%")
        else:
            degradation = None
        
        status = 'PASS' if passed else 'FAIL'
        
        print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: {strategy_name}")
        if not passed:
            for reason in fail_reasons:
                print(f"   ⚠️ {reason}")
        
        return {
            'strategy': strategy_name,
            'status': status,
            'in_sample': is_result,
            'out_of_sample': oos_result,
            'degradation': degradation,
            'fail_reasons': fail_reasons,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_rolling_window(
        self,
        strategy_name: str,
        data: pd.DataFrame
    ) -> Dict:
        """
        Run rolling window Walk-Forward Analysis.
        
        Runs multiple windows and aggregates results for robust validation.
        """
        strategy_class, preprocess_func, error = self.discover_strategy_class(strategy_name)
        if error:
            return {'strategy': strategy_name, 'status': 'ERROR', 'error': error}
        
        total_rows = len(data)
        window_size = int(total_rows * (self.in_sample_ratio + self.out_of_sample_ratio * 0.5))
        step_size = int(total_rows * self.rolling_step_ratio)
        
        print(f"\n{'='*60}")
        print(f"🔄 Rolling Window WFA: {strategy_name}")
        print(f"{'='*60}")
        print(f"📊 Total rows: {total_rows}")
        print(f"🪟 Window size: {window_size} rows")
        print(f"👣 Step size: {step_size} rows")
        print(f"🔢 Windows: {self.rolling_windows}")
        
        if preprocess_func:
            print(f"📦 Using preprocessing function")
        
        window_results = []
        
        for i in range(self.rolling_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx > total_rows:
                print(f"\n⚠️ Window {i+1} would exceed data, stopping")
                break
            
            window_data = data.iloc[start_idx:end_idx].copy()
            split_idx = int(len(window_data) * self.in_sample_ratio)
            
            is_data = window_data.iloc[:split_idx]
            oos_data = window_data.iloc[split_idx:]
            
            print(f"\n📍 Window {i+1}: rows {start_idx}-{end_idx}")
            
            is_result = self._run_backtest(strategy_class, is_data, f"window_{i+1}_is", preprocess_func)
            oos_result = self._run_backtest(strategy_class, oos_data, f"window_{i+1}_oos", preprocess_func)
            
            is_return = is_result.get('return_pct', 0) or 0
            oos_return = oos_result.get('return_pct', 0) or 0
            
            print(f"   IS: {is_return:+.2f}% | OOS: {oos_return:+.2f}%")
            
            window_results.append({
                'window': i + 1,
                'in_sample': is_result,
                'out_of_sample': oos_result,
                'is_return': is_return,
                'oos_return': oos_return
            })
        
        # Aggregate results
        oos_returns = [w['oos_return'] for w in window_results]
        avg_oos_return = np.mean(oos_returns) if oos_returns else 0
        positive_windows = sum(1 for r in oos_returns if r > 0)
        
        # Pass if majority of windows are profitable
        passed = positive_windows >= len(window_results) / 2 and avg_oos_return > 0
        
        print(f"\n{'='*60}")
        print(f"📊 Rolling Window Summary")
        print(f"   Avg OOS Return: {avg_oos_return:+.2f}%")
        print(f"   Profitable Windows: {positive_windows}/{len(window_results)}")
        print(f"   {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return {
            'strategy': strategy_name,
            'status': 'PASS' if passed else 'FAIL',
            'windows': window_results,
            'avg_oos_return': avg_oos_return,
            'positive_windows': positive_windows,
            'total_windows': len(window_results),
            'timestamp': datetime.now().isoformat()
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Walk-Forward Analysis')
    parser.add_argument('--strategy', type=str, required=True,
                        help='Strategy name to analyze')
    parser.add_argument('--dataset', type=str, default='BTC_15m',
                        help='Dataset name (default: BTC_15m)')
    parser.add_argument('--rolling', action='store_true',
                        help='Use rolling window instead of single split')
    parser.add_argument('--windows', type=int, default=3,
                        help='Number of rolling windows (default: 3)')
    parser.add_argument('--split', type=float, default=0.70,
                        help='In-sample ratio (default: 0.70)')
    args = parser.parse_args()
    
    analyzer = WalkForwardAnalyzer(
        in_sample_ratio=args.split,
        rolling_windows=args.windows
    )
    
    # Load data
    data = analyzer.load_data(args.dataset)
    if data is None:
        sys.exit(1)
    
    # Run analysis
    if args.rolling:
        result = analyzer.run_rolling_window(args.strategy, data)
    else:
        result = analyzer.run_single_split(args.strategy, data)
    
    # Save result
    output_file = RESULTS_DIR / f"wfa_{args.strategy}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {output_file}")
    
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
