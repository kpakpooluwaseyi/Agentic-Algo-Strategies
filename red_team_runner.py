#!/usr/bin/env python3
"""
🔴 Red Team Runner: Comprehensive Strategy Validation Pipeline
===============================================================

Orchestrates the full red team validation process:
1. Find strategies with positive ROI from leaderboards
2. Stage 1: Single 70/30 Walk-Forward Analysis
3. Stage 2: Rolling Window WFA (for Stage 1 graduates)
4. Stage 3: Monte Carlo stress testing (for Stage 2 graduates)
5. Generate comprehensive report

Usage:
    python red_team_runner.py --dataset BTC_15m
    python red_team_runner.py --dataset BTC_15m --strategy silver_bullet_fvg_retest
    python red_team_runner.py --all --verbose
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from walk_forward import WalkForwardAnalyzer
from red_team_stress_test import RedTeamTester, AdversarialDataGenerator, STRATEGY_TYPE_MAP

# Paths
RESULTS_DIR = Path(__file__).parent / 'results'
LEADERBOARDS_DIR = RESULTS_DIR / 'leaderboards'
RED_TEAM_DIR = RESULTS_DIR / 'red_team'
DATA_DIR = Path(__file__).parent / 'data'


class RedTeamRunner:
    """
    Orchestrates the complete red team validation pipeline.
    
    Pipeline:
    1. Identify positive ROI strategies from leaderboards
    2. Run single 70/30 WFA - filter out overfitted strategies
    3. Run rolling window WFA on graduates - more robust validation
    4. Run Monte Carlo stress tests on final graduates
    5. Generate comprehensive markdown report
    """
    
    def __init__(self, dataset: str = 'BTC_15m', verbose: bool = False):
        self.dataset = dataset
        self.verbose = verbose
        self.wfa_analyzer = WalkForwardAnalyzer()
        self.mc_tester = RedTeamTester(output_dir=str(RED_TEAM_DIR))
        self.generator = AdversarialDataGenerator()
        
        RED_TEAM_DIR.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.positive_roi_strategies: List[Dict] = []
        self.stage1_results: List[Dict] = []
        self.stage2_results: List[Dict] = []
        self.stage3_results: List[Dict] = []
    
    def get_positive_roi_strategies(self) -> List[Dict]:
        """
        Parse leaderboard CSVs to find strategies with positive returns.
        
        Returns list of dicts with strategy_name and return_pct.
        """
        print("\n" + "=" * 60)
        print("📊 Stage 0: Finding Positive ROI Strategies")
        print("=" * 60)
        
        strategies = {}
        
        # Look for leaderboard files
        leaderboard_dirs = list(LEADERBOARDS_DIR.glob('*/'))
        
        if not leaderboard_dirs:
            # Fallback to main leaderboard.csv
            main_leaderboard = RESULTS_DIR / 'leaderboard.csv'
            if main_leaderboard.exists():
                df = pd.read_csv(main_leaderboard)
                for _, row in df.iterrows():
                    name = row.get('strategy_name')
                    ret = row.get('return_pct', 0)
                    if name and ret and ret > 0:
                        if name not in strategies or ret > strategies[name]:
                            strategies[name] = ret
        
        # Parse all leaderboard files
        for lb_dir in leaderboard_dirs:
            lb_file = lb_dir / 'leaderboard.csv'
            if lb_file.exists():
                try:
                    df = pd.read_csv(lb_file)
                    for _, row in df.iterrows():
                        name = row.get('strategy_name')
                        ret = row.get('return_pct')
                        status = row.get('status')
                        
                        if name and status == 'SUCCESS' and ret is not None and ret > 0:
                            # Keep best return for each strategy
                            if name not in strategies or ret > strategies[name]:
                                strategies[name] = ret
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠️ Error reading {lb_file}: {e}")
        
        # Convert to list and sort by return
        self.positive_roi_strategies = [
            {'strategy_name': name, 'return_pct': ret}
            for name, ret in sorted(strategies.items(), key=lambda x: x[1], reverse=True)
        ]
        
        print(f"✅ Found {len(self.positive_roi_strategies)} strategies with positive ROI")
        
        if self.verbose and self.positive_roi_strategies:
            print("\n🏆 Top 10 Positive ROI Strategies:")
            for i, s in enumerate(self.positive_roi_strategies[:10], 1):
                print(f"   {i}. {s['strategy_name']}: +{s['return_pct']:.2f}%")
        
        return self.positive_roi_strategies
    
    def run_stage1_wfa(self, strategies: Optional[List[str]] = None) -> List[Dict]:
        """
        Stage 1: Single 70/30 Walk-Forward Analysis.
        
        Filters out strategies that fail basic out-of-sample validation.
        """
        print("\n" + "=" * 60)
        print("🔬 Stage 1: Single Split Walk-Forward Analysis (70/30)")
        print("=" * 60)
        
        if strategies is None:
            strategies = [s['strategy_name'] for s in self.positive_roi_strategies]
        
        # Load data
        data = self.wfa_analyzer.load_data(self.dataset)
        if data is None:
            print("❌ Failed to load dataset")
            return []
        
        passed = []
        failed = []
        
        for strategy_name in strategies:
            result = self.wfa_analyzer.run_single_split(strategy_name, data)
            self.stage1_results.append(result)
            
            if result['status'] == 'PASS':
                passed.append(result)
            else:
                failed.append(result)
        
        print(f"\n📈 Stage 1 Summary:")
        print(f"   ✅ Passed: {len(passed)}")
        print(f"   ❌ Failed: {len(failed)}")
        
        return passed
    
    def run_stage2_rolling(self, strategies: List[str]) -> List[Dict]:
        """
        Stage 2: Rolling Window Walk-Forward Analysis.
        
        More robust validation for strategies that passed Stage 1.
        """
        print("\n" + "=" * 60)
        print("🔄 Stage 2: Rolling Window Walk-Forward Analysis")
        print("=" * 60)
        
        if not strategies:
            print("⚠️ No strategies to test in Stage 2")
            return []
        
        # Load data
        data = self.wfa_analyzer.load_data(self.dataset)
        if data is None:
            print("❌ Failed to load dataset")
            return []
        
        passed = []
        failed = []
        
        for strategy_name in strategies:
            result = self.wfa_analyzer.run_rolling_window(strategy_name, data)
            self.stage2_results.append(result)
            
            if result['status'] == 'PASS':
                passed.append(result)
            else:
                failed.append(result)
        
        print(f"\n📈 Stage 2 Summary:")
        print(f"   ✅ Passed: {len(passed)}")
        print(f"   ❌ Failed: {len(failed)}")
        
        return passed
    
    def run_stage3_monte_carlo(self, strategies: List[str]) -> List[Dict]:
        """
        Stage 3: Monte Carlo Stress Testing.
        
        Tests strategies against adversarial market conditions.
        """
        print("\n" + "=" * 60)
        print("🎲 Stage 3: Monte Carlo Stress Testing")
        print("=" * 60)
        
        if not strategies:
            print("⚠️ No strategies to test in Stage 3")
            return []
        
        # For each strategy, run all applicable scenarios
        for strategy_name in strategies:
            print(f"\n🔴 Testing: {strategy_name}")
            
            # Determine strategy type
            strategy_type = STRATEGY_TYPE_MAP.get(strategy_name, 'session')  # default to session
            
            result = {
                'strategy': strategy_name,
                'strategy_type': strategy_type,
                'scenarios': []
            }
            
            # Run standard scenarios
            scenarios = [
                ('WHIPSAW_CHOP', self.generator.generate_whipsaw_chop()),
                ('EXTENDED_TREND_UP', self.generator.generate_extended_trend(direction='up')),
                ('EXTENDED_TREND_DOWN', self.generator.generate_extended_trend(direction='down')),
                ('FALSE_BREAKOUTS', self.generator.generate_false_breakouts()),
                ('VOLATILITY_EXPLOSION', self.generator.generate_volatility_explosion()),
            ]
            
            for scenario_name, data in scenarios:
                try:
                    # Import and run strategy
                    strategy_class, error = self.wfa_analyzer.discover_strategy_class(strategy_name)
                    if error:
                        result['scenarios'].append({
                            'scenario': scenario_name,
                            'status': 'ERROR',
                            'error': error
                        })
                        continue
                    
                    from backtesting import Backtest
                    bt = Backtest(data, strategy_class, cash=100_000, commission=0.002)
                    stats = bt.run()
                    
                    ret = float(stats.get('Return [%]', 0)) if stats.get('Return [%]') else 0
                    trades = int(stats.get('# Trades', 0)) if stats.get('# Trades') else 0
                    
                    scenario_result = {
                        'scenario': scenario_name,
                        'return_pct': ret,
                        'trades': trades,
                        'status': 'SURVIVED' if ret > -30 else 'FAILED'
                    }
                    result['scenarios'].append(scenario_result)
                    
                    status_icon = '✅' if ret > -30 else '❌'
                    print(f"   {status_icon} {scenario_name}: {ret:+.2f}% ({trades} trades)")
                    
                except Exception as e:
                    result['scenarios'].append({
                        'scenario': scenario_name,
                        'status': 'ERROR',
                        'error': str(e)[:100]
                    })
            
            self.stage3_results.append(result)
        
        return self.stage3_results
    
    def generate_report(self) -> str:
        """Generate comprehensive markdown report."""
        report_path = RED_TEAM_DIR / 'summary_report.md'
        
        # Count results
        s1_passed = [r for r in self.stage1_results if r.get('status') == 'PASS']
        s2_passed = [r for r in self.stage2_results if r.get('status') == 'PASS']
        
        report = f"""# 🔴 Red Team Validation Report

**Dataset:** {self.dataset}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Stage | Strategies Tested | Passed | Pass Rate |
|-------|-------------------|--------|-----------|
| Initial Pool (Positive ROI) | {len(self.positive_roi_strategies)} | - | - |
| Stage 1: Single 70/30 WFA | {len(self.stage1_results)} | {len(s1_passed)} | {len(s1_passed)/max(len(self.stage1_results),1)*100:.0f}% |
| Stage 2: Rolling Window WFA | {len(self.stage2_results)} | {len(s2_passed)} | {len(s2_passed)/max(len(self.stage2_results),1)*100:.0f}% |
| Stage 3: Monte Carlo | {len(self.stage3_results)} | - | - |

---

## 🏆 Validated Strategies

Strategies that passed ALL validation stages:

"""
        # Find strategies that passed all stages
        s2_passed_names = {r['strategy'] for r in s2_passed}
        
        if s2_passed_names:
            for name in s2_passed_names:
                # Get OOS return from stage 1
                s1_result = next((r for r in self.stage1_results if r['strategy'] == name), {})
                oos_return = s1_result.get('out_of_sample', {}).get('return_pct', 'N/A')
                report += f"- **{name}**: OOS Return {oos_return}%\n"
        else:
            report += "*No strategies passed all validation stages.*\n"
        
        # Stage 1 Details
        report += f"\n---\n\n## Stage 1: Single Split WFA Results\n\n"
        report += "| Strategy | IS Return | OOS Return | Trades | Status |\n"
        report += "|----------|-----------|------------|--------|--------|\n"
        
        for r in self.stage1_results:
            is_ret = r.get('in_sample', {}).get('return_pct', 'N/A')
            oos_ret = r.get('out_of_sample', {}).get('return_pct', 'N/A')
            trades = r.get('out_of_sample', {}).get('trades', 'N/A')
            status = '✅' if r.get('status') == 'PASS' else '❌'
            report += f"| {r['strategy']} | {is_ret} | {oos_ret} | {trades} | {status} |\n"
        
        # Stage 2 Details
        if self.stage2_results:
            report += f"\n---\n\n## Stage 2: Rolling Window WFA Results\n\n"
            report += "| Strategy | Avg OOS Return | Profitable Windows | Status |\n"
            report += "|----------|----------------|-------------------|--------|\n"
            
            for r in self.stage2_results:
                avg_oos = r.get('avg_oos_return', 'N/A')
                pos_wins = f"{r.get('positive_windows', 0)}/{r.get('total_windows', 0)}"
                status = '✅' if r.get('status') == 'PASS' else '❌'
                report += f"| {r['strategy']} | {avg_oos:.2f}% | {pos_wins} | {status} |\n"
        
        # Stage 3 Details
        if self.stage3_results:
            report += f"\n---\n\n## Stage 3: Monte Carlo Stress Test Results\n\n"
            for r in self.stage3_results:
                report += f"\n### {r['strategy']}\n\n"
                report += "| Scenario | Return | Trades | Status |\n"
                report += "|----------|--------|--------|--------|\n"
                for s in r.get('scenarios', []):
                    ret = s.get('return_pct', 'N/A')
                    trades = s.get('trades', 'N/A')
                    status = '✅' if s.get('status') == 'SURVIVED' else '❌'
                    report += f"| {s['scenario']} | {ret}% | {trades} | {status} |\n"
        
        report += f"\n---\n\n*Report generated by Red Team Runner*\n"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📋 Report saved to: {report_path}")
        return str(report_path)
    
    def run_full_pipeline(self, strategy_filter: Optional[str] = None):
        """Run the complete red team validation pipeline."""
        print("\n" + "=" * 70)
        print("🔴 RED TEAM VALIDATION PIPELINE")
        print("=" * 70)
        print(f"Dataset: {self.dataset}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Stage 0: Find positive ROI strategies
        self.get_positive_roi_strategies()
        
        if not self.positive_roi_strategies:
            print("\n❌ No positive ROI strategies found!")
            return
        
        # Apply filter if specified
        if strategy_filter:
            strategies_to_test = [
                s['strategy_name'] for s in self.positive_roi_strategies
                if strategy_filter.lower() in s['strategy_name'].lower()
            ]
        else:
            strategies_to_test = [s['strategy_name'] for s in self.positive_roi_strategies]
        
        print(f"\n📋 Testing {len(strategies_to_test)} strategies")
        
        # Stage 1: Single 70/30 WFA
        stage1_passed = self.run_stage1_wfa(strategies_to_test)
        stage1_names = [r['strategy'] for r in stage1_passed]
        
        # Stage 2: Rolling Window WFA (only for Stage 1 graduates)
        if stage1_names:
            stage2_passed = self.run_stage2_rolling(stage1_names)
            stage2_names = [r['strategy'] for r in stage2_passed]
        else:
            stage2_names = []
        
        # Stage 3: Monte Carlo (only for Stage 2 graduates)
        if stage2_names:
            self.run_stage3_monte_carlo(stage2_names)
        
        # Generate report
        self.generate_report()
        
        # Save raw results
        results_file = RED_TEAM_DIR / 'pipeline_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'dataset': self.dataset,
                'timestamp': datetime.now().isoformat(),
                'positive_roi_strategies': self.positive_roi_strategies,
                'stage1_results': self.stage1_results,
                'stage2_results': self.stage2_results,
                'stage3_results': self.stage3_results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Raw results saved to: {results_file}")
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎉 RED TEAM VALIDATION COMPLETE")
        print("=" * 70)
        s2_passed = [r for r in self.stage2_results if r.get('status') == 'PASS']
        print(f"✅ Strategies that passed all validation: {len(s2_passed)}")
        for r in s2_passed:
            print(f"   - {r['strategy']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Red Team Validation Pipeline')
    parser.add_argument('--dataset', type=str, default='BTC_15m',
                        help='Dataset name (default: BTC_15m)')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Filter to specific strategy (partial match)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--all', action='store_true',
                        help='Run on all positive ROI strategies')
    args = parser.parse_args()
    
    runner = RedTeamRunner(dataset=args.dataset, verbose=args.verbose)
    runner.run_full_pipeline(strategy_filter=args.strategy)


if __name__ == '__main__':
    main()
