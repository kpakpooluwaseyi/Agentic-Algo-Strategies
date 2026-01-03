#!/usr/bin/env python3
"""
📊 Strategy Dashboard Generator
Creates a markdown report showing each strategy's journey through:
  Backtest → Walk-Forward Analysis → Red Team

Usage:
    python strategy_dashboard.py > results/strategy_report.md
    python strategy_dashboard.py --json > results/strategy_report.json
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Paths
RESULTS_DIR = Path('results')
STRATEGIES_DIR = Path('strategies')
MERGED_LEADERBOARD = RESULTS_DIR / 'leaderboard_merged.csv'
CURRENT_LEADERBOARD = RESULTS_DIR / 'leaderboard.csv'
RED_TEAM_DIR = RESULTS_DIR / 'red_team'
OUTPUT_MD = RESULTS_DIR / 'strategy_report.md'


def load_leaderboard() -> pd.DataFrame:
    """Load the best available leaderboard."""
    if MERGED_LEADERBOARD.exists():
        df = pd.read_csv(MERGED_LEADERBOARD)
    elif CURRENT_LEADERBOARD.exists():
        df = pd.read_csv(CURRENT_LEADERBOARD)
    else:
        return pd.DataFrame()
    
    # Ensure numeric columns
    numeric_cols = ['return_pct', 'sharpe_ratio', 'max_drawdown_pct', 
                    'win_rate_pct', 'total_trades', 'wfa_oos_return_pct', 'wfa_degradation_pct']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_red_team_results() -> dict:
    """Load red team WFA results."""
    results = {}
    if RED_TEAM_DIR.exists():
        for json_file in RED_TEAM_DIR.glob('wfa_*.json'):
            try:
                strategy_name = json_file.stem.replace('wfa_', '')
                with open(json_file, 'r') as f:
                    results[strategy_name] = json.load(f)
            except Exception:
                pass
    return results


def get_all_strategies() -> list:
    """Get all strategy names from strategies folder."""
    strategies = []
    for py_file in STRATEGIES_DIR.glob('*.py'):
        if not py_file.stem.startswith('__'):
            strategies.append(py_file.stem)
    return sorted(strategies)


def determine_failure_point(row: pd.Series, red_team: dict) -> str:
    """Determine where a strategy failed in the pipeline."""
    strategy = row.get('strategy_name', '')
    
    # Check backtest
    if row.get('status') == 'ERROR':
        return '❌ Backtest: Execution Error'
    
    return_pct = row.get('return_pct')
    if pd.isna(return_pct):
        return '⚠️ Backtest: No Trades'
    
    if return_pct <= 0:
        return f'❌ Backtest: Negative Return ({return_pct:.1f}%)'
    
    # Check WFA
    wfa_status = row.get('wfa_status')
    if wfa_status == 'FAIL':
        oos_return = row.get('wfa_oos_return_pct')
        degradation = row.get('wfa_degradation_pct')
        if pd.notna(degradation) and degradation > 0.5:
            return f'❌ WFA: High Degradation ({degradation*100:.0f}%)'
        if pd.notna(oos_return) and oos_return <= 0:
            return f'❌ WFA: OOS Negative ({oos_return:.1f}%)'
        return '❌ WFA: Failed Validation'
    
    if wfa_status == 'ERROR':
        return '⚠️ WFA: Error'
    
    if wfa_status == 'SKIPPED':
        return '⏭️ WFA: Not Run Yet'
    
    # Check Red Team
    if strategy in red_team:
        rt = red_team[strategy]
        if rt.get('status') == 'FAIL':
            reasons = rt.get('fail_reasons', [])
            if reasons:
                return f"❌ Red Team: {reasons[0]}"
            return '❌ Red Team: Failed'
    
    # Passed so far
    if wfa_status == 'PASS':
        if strategy in red_team and red_team[strategy].get('status') == 'PASS':
            return '✅ All Stages Passed'
        return '✅ Backtest + WFA Passed (Red Team Pending)'
    
    return '⏸️ In Progress'


def generate_markdown_report(df: pd.DataFrame, red_team: dict, all_strategies: list) -> str:
    """Generate markdown report."""
    lines = [
        "# 📊 Strategy Performance Dashboard",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total Strategies | {len(all_strategies)} |",
        f"| Tested | {df['strategy_name'].nunique() if not df.empty else 0} |",
        f"| Not Yet Tested | {len(all_strategies) - (df['strategy_name'].nunique() if not df.empty else 0)} |",
        "",
    ]
    
    if df.empty:
        lines.append("⚠️ No leaderboard data found. Run `python merge_leaderboards.py` first.")
        return '\n'.join(lines)
    
    # Aggregate by strategy (best dataset performance)
    strategy_summary = []
    tested_strategies = df['strategy_name'].unique()
    
    for strategy in all_strategies:
        if strategy in tested_strategies:
            strat_df = df[df['strategy_name'] == strategy]
            best_row = strat_df.loc[strat_df['return_pct'].idxmax()] if strat_df['return_pct'].notna().any() else strat_df.iloc[0]
            
            strategy_summary.append({
                'strategy': strategy,
                'best_return': best_row.get('return_pct'),
                'best_dataset': best_row.get('dataset_name', 'N/A'),
                'sharpe': best_row.get('sharpe_ratio'),
                'trades': best_row.get('total_trades'),
                'wfa_status': best_row.get('wfa_status', 'N/A'),
                'failure_point': determine_failure_point(best_row, red_team),
                'status': best_row.get('status', 'N/A')
            })
        else:
            strategy_summary.append({
                'strategy': strategy,
                'best_return': None,
                'best_dataset': 'N/A',
                'sharpe': None,
                'trades': None,
                'wfa_status': 'NOT RUN',
                'failure_point': '⏸️ Not Tested Yet',
                'status': 'NOT RUN'
            })
    
    # Sort by return (best first, None at bottom)
    strategy_summary.sort(key=lambda x: (x['best_return'] is None, -(x['best_return'] or -999)))
    
    # Top Performers
    lines.extend([
        "## 🏆 Top Performers (Positive Returns)",
        "",
        "| Strategy | Best Return | Dataset | Sharpe | Trades | WFA | Status |",
        "|----------|-------------|---------|--------|--------|-----|--------|",
    ])
    
    for s in strategy_summary:
        if s['best_return'] is not None and s['best_return'] > 0:
            ret = f"+{s['best_return']:.2f}%" if s['best_return'] else 'N/A'
            sharpe = f"{s['sharpe']:.2f}" if pd.notna(s['sharpe']) else '-'
            trades = int(s['trades']) if pd.notna(s['trades']) else '-'
            wfa = s['wfa_status'] or 'N/A'
            lines.append(f"| `{s['strategy']}` | {ret} | {s['best_dataset']} | {sharpe} | {trades} | {wfa} | {s['failure_point']} |")
    
    # Negative Returns
    lines.extend([
        "",
        "## 📉 Negative Returns (Potential for Reversal Strategy?)",
        "",
        "| Strategy | Return | Dataset | Failure Point |",
        "|----------|--------|---------|---------------|",
    ])
    
    for s in strategy_summary:
        if s['best_return'] is not None and s['best_return'] < 0:
            ret = f"{s['best_return']:.2f}%"
            lines.append(f"| `{s['strategy']}` | {ret} | {s['best_dataset']} | {s['failure_point']} |")
    
    # Errors and Not Tested
    lines.extend([
        "",
        "## ⚠️ Errors & Not Tested",
        "",
        "| Strategy | Status | Notes |",
        "|----------|--------|-------|",
    ])
    
    for s in strategy_summary:
        if s['best_return'] is None:
            lines.append(f"| `{s['strategy']}` | {s['status']} | {s['failure_point']} |")
    
    return '\n'.join(lines)


def main():
    import sys
    
    print("📊 Generating Strategy Dashboard...", file=sys.stderr)
    
    df = load_leaderboard()
    red_team = load_red_team_results()
    all_strategies = get_all_strategies()
    
    print(f"  Loaded {len(df)} leaderboard rows", file=sys.stderr)
    print(f"  Found {len(red_team)} red team results", file=sys.stderr)
    print(f"  {len(all_strategies)} strategies in folder", file=sys.stderr)
    
    if '--json' in sys.argv:
        # JSON output
        output = {
            'generated': datetime.now().isoformat(),
            'total_strategies': len(all_strategies),
            'tested': df['strategy_name'].nunique() if not df.empty else 0,
            'strategies': all_strategies
        }
        print(json.dumps(output, indent=2))
    else:
        # Markdown output
        report = generate_markdown_report(df, red_team, all_strategies)
        print(report)
        
        # Also save to file
        with open(OUTPUT_MD, 'w') as f:
            f.write(report)
        print(f"\n✅ Saved to: {OUTPUT_MD}", file=sys.stderr)


if __name__ == '__main__':
    main()
