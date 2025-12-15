#!/usr/bin/env python3
"""
Batch updates all strategy files to add dual-mode support.
This allows strategies to use either:
1. External data via BACKTEST_DATA_PATH environment variable
2. Original synthetic data generation (standalone mode)
"""

import re
from pathlib import Path

STRATEGIES_DIR = Path(__file__).parent / 'strategies'

# Strategies that already have dual-mode support
ALREADY_UPDATED = {
    '50_50_mow_internal_scalp',
    'asia_liquidity_reversal_uk_session', 
    'asia_range_liquidity_sweep_uk_reversal',
    'ict_asian_sweep_london_open',
    'predictable_candle_expansion_displacement_model',
    'session_liquidity_grab_reversal'
}

def find_strategy_class(content: str) -> str:
    """Extract the Strategy class name from the file."""
    match = re.search(r'class\s+(\w+)\s*\(\s*Strategy\s*\)', content)
    return match.group(1) if match else None

def find_main_block(content: str):
    """Find the __main__ block start line."""
    match = re.search(r"if __name__\s*==\s*['\"]__main__['\"]\s*:", content)
    if match:
        return match.start()
    return None

def update_strategy(filepath: Path, dry_run: bool = True) -> bool:
    """Update a strategy file with dual-mode support."""
    content = filepath.read_text()
    
    # Skip if already has dual-mode
    if 'BACKTEST_DATA_PATH' in content:
        print(f"  ⏭️  {filepath.stem}: Already has dual-mode support")
        return False
    
    # Find strategy class name
    strategy_class = find_strategy_class(content)
    if not strategy_class:
        print(f"  ❌ {filepath.stem}: Could not find Strategy class")
        return False
    
    # Find __main__ block
    main_start = find_main_block(content)
    if main_start is None:
        print(f"  ❌ {filepath.stem}: No __main__ block found")
        return False
    
    # Get the existing main block content
    main_content = content[main_start:]
    
    # Check if there's a generate_synthetic_data function
    has_synthetic = 'generate_synthetic_data' in content or 'def generate_' in content
    
    # Check if there's a preprocess_data function (module level, not in __main__)
    has_preprocess = re.search(r'^def preprocess_data\(', content, re.MULTILINE)
    
    # Create the new __main__ block
    new_main = f'''if __name__ == '__main__':
    import os
    import json
    from backtesting import Backtest
    
    data_path = os.environ.get('BACKTEST_DATA_PATH')
    mode = os.environ.get('BACKTEST_MODE', 'standalone')
    
    if data_path and os.path.exists(data_path):
        # === STANDARDIZED MODE ===
        print(f"[Standardized Mode] Loading data from: {{data_path}}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.title() for c in data.columns]
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
'''
    
    # Add preprocessing if function exists at module level
    if has_preprocess:
        new_main += '''        
        # Apply preprocessing
        try:
            data = preprocess_data(data)
        except Exception as e:
            print(f"Preprocessing warning: {e}")
'''
    
    new_main += f'''        
        from backtesting.lib import FractionalBacktest
        bt = FractionalBacktest(data, {strategy_class}, cash=1_000_000, commission=.002, fractional_unit=1e-4)
        
        # In standardized mode, always run with defaults (optimization requires strategy-specific params)
        print("[Run Mode] Running single backtest with defaults...")
        stats = bt.run()
        
        # Save results
        os.makedirs('results', exist_ok=True)
        result = {{
            'strategy_name': '{filepath.stem}',
            'return': float(stats.get('Return [%]', 0)) if not pd.isna(stats.get('Return [%]', 0)) else None,
            'sharpe': float(stats.get('Sharpe Ratio')) if stats.get('Sharpe Ratio') and not pd.isna(stats.get('Sharpe Ratio')) else None,
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)) if not pd.isna(stats.get('Max. Drawdown [%]', 0)) else None,
            'win_rate': float(stats.get('Win Rate [%]', 0)) if not pd.isna(stats.get('Win Rate [%]', 0)) else None,
            'total_trades': int(stats.get('# Trades', 0))
        }}
        with open('results/temp_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Return={{result['return']}}%, Trades={{result['total_trades']}}")
    else:
        # === STANDALONE MODE (original behavior) ===
        print("[Standalone Mode] Using original data generation...")
'''
    
    # Now we need to preserve the original __main__ content as the standalone fallback
    # Extract just the body of the original __main__ block
    original_body_lines = main_content.split('\n')[1:]  # Skip the 'if __name__' line
    
    # Re-indent the original body to be inside the else block
    # Original code has 4-space indent (inside __main__), we add 4 more for the else block = 8 total
    indented_lines = []
    for line in original_body_lines:
        if line.strip():  # Non-empty lines
            # Add 4 spaces to the BEGINNING of the line (preserving existing indent)
            indented_lines.append('    ' + line)
        else:
            indented_lines.append('')
    
    new_main += '\n'.join(indented_lines)
    
    # Replace the old __main__ block with the new one
    new_content = content[:main_start] + new_main
    
    if dry_run:
        print(f"  🔍 {filepath.stem}: Would update (class={strategy_class}, preprocess={bool(has_preprocess)})")
        return True
    else:
        filepath.write_text(new_content)
        print(f"  ✅ {filepath.stem}: Updated successfully")
        return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch update strategies with dual-mode support')
    parser.add_argument('--apply', action='store_true', help='Actually apply changes (default is dry-run)')
    parser.add_argument('--strategy', type=str, help='Update only a specific strategy')
    args = parser.parse_args()
    
    dry_run = not args.apply
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be changed")
        print("   Use --apply to actually update files\n")
    else:
        print("⚠️  APPLYING CHANGES - Files will be modified\n")
    
    strategies = sorted(STRATEGIES_DIR.glob('*.py'))
    
    if args.strategy:
        strategies = [s for s in strategies if args.strategy.lower() in s.stem.lower()]
    
    updated = 0
    skipped = 0
    failed = 0
    
    for filepath in strategies:
        if filepath.stem.startswith('__'):
            continue
        if filepath.stem in ALREADY_UPDATED:
            print(f"  ⏭️  {filepath.stem}: Already manually updated")
            skipped += 1
            continue
            
        try:
            if update_strategy(filepath, dry_run=dry_run):
                updated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ❌ {filepath.stem}: Error - {e}")
            failed += 1
    
    print(f"\n📊 Summary: {updated} updated, {skipped} skipped, {failed} failed")
    
    if dry_run and updated > 0:
        print("\n💡 Run with --apply to actually update the files")

if __name__ == '__main__':
    main()
