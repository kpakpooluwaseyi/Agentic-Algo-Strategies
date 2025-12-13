#!/usr/bin/env python3
"""
Batch update strategies to support dual-mode execution.
Uses AST to properly parse and modify Python files.
"""

import os
import re
import ast
from pathlib import Path

STRATEGIES_DIR = Path('strategies')

def get_strategy_class_name(filepath: Path) -> str:
    """Find the main Strategy class name using AST"""
    content = filepath.read_text()
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'Strategy':
                        return node.name
    except:
        pass
    return None


def get_main_block_range(content: str) -> tuple:
    """Find the line range of __main__ block"""
    lines = content.split('\n')
    start_line = None
    
    for i, line in enumerate(lines):
        if "__name__" in line and "__main__" in line:
            start_line = i
            break
    
    if start_line is None:
        return None, None
    
    # __main__ goes to end of file
    return start_line, len(lines)


def get_optimize_params(content: str) -> str:
    """Extract optimization parameters from bt.optimize() call"""
    # Find the bt.optimize call and extract everything between parentheses
    # Handle multiline by finding matching parens
    match = re.search(r'bt\.optimize\s*\(', content)
    if not match:
        return None
    
    start = match.end()
    paren_depth = 1
    end = start
    
    while end < len(content) and paren_depth > 0:
        if content[end] == '(':
            paren_depth += 1
        elif content[end] == ')':
            paren_depth -= 1
        end += 1
    
    params = content[start:end-1].strip()
    return params


def get_data_generation_call(content: str) -> str:
    """Find how data is loaded (generate_synthetic_data or similar)"""
    # Look for data = ... patterns
    patterns = [
        (r'data\s*=\s*generate_synthetic_data\([^)]*\)', 'generate_synthetic_data()'),
        (r'raw_data\s*=\s*generate_synthetic_data\([^)]*\)', 'generate_synthetic_data()'),
        (r'data\s*=\s*GOOG\.copy\(\)', 'GOOG.copy()'),
    ]
    
    for pattern, default in patterns:
        if re.search(pattern, content):
            match = re.search(pattern, content)
            return match.group(0).split('=')[1].strip()
    
    return 'generate_synthetic_data()'


def create_new_main_block(strategy_class: str, strategy_name: str, 
                         optimize_params: str, data_gen_call: str,
                         has_preprocess: bool = False) -> str:
    """Create the new dual-mode __main__ block"""
    
    # Build the optimize call
    if optimize_params:
        optimize_call = f"stats = bt.optimize(\n            {optimize_params}\n        )"
    else:
        optimize_call = "stats = bt.run()"
    
    # Data loading for standalone mode
    if 'preprocess' in data_gen_call.lower() or has_preprocess:
        standalone_data = f"data = {data_gen_call}"
    else:
        standalone_data = f"data = {data_gen_call}"
    
    return f'''if __name__ == '__main__':
    import os
    
    # === DUAL MODE SUPPORT ===
    data_path = os.environ.get('BACKTEST_DATA_PATH')
    mode = os.environ.get('BACKTEST_MODE', 'standalone')
    
    if data_path and os.path.exists(data_path):
        # --- STANDARDIZED MODE ---
        print(f"[Standardized Mode] Loading: {{data_path}}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.title() for c in data.columns]
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        bt = Backtest(data, {strategy_class}, cash=10000, commission=.002)
        
        if mode == 'optimize':
            print("[Optimize Mode] Running optimization...")
            {optimize_call}
        else:
            print("[Run Mode] Single backtest...")
            stats = bt.run()
    else:
        # --- STANDALONE MODE ---
        print("[Standalone Mode] Using built-in data...")
        {standalone_data}
        bt = Backtest(data, {strategy_class}, cash=10000, commission=.002)
        {optimize_call}
        try:
            bt.plot(filename='results/{strategy_name}.html')
        except Exception as e:
            print(f"Plot error: {{e}}")
    
    # --- SAVE RESULTS ---
    os.makedirs('results', exist_ok=True)
    
    def _sanitize(v):
        import numpy as np
        if isinstance(v, (np.integer, np.int64)): return int(v)
        if isinstance(v, (np.floating, np.float64)): 
            return None if np.isnan(v) else float(v)
        if isinstance(v, (pd.Series, pd.DataFrame)): return None
        if v is None or (isinstance(v, float) and np.isnan(v)): return None
        return v
    
    result = {{
        'strategy_name': '{strategy_name}',
        'return': _sanitize(stats.get('Return [%]')),
        'sharpe': _sanitize(stats.get('Sharpe Ratio')),
        'max_drawdown': _sanitize(stats.get('Max. Drawdown [%]')),
        'win_rate': _sanitize(stats.get('Win Rate [%]')),
        'total_trades': _sanitize(stats.get('# Trades', 0))
    }}
    
    with open('results/temp_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Return={{result['return']}}, Trades={{result['total_trades']}}")
'''


def update_strategy(filepath: Path, dry_run: bool = True) -> tuple:
    """Update a strategy file. Returns (success, message)"""
    content = filepath.read_text()
    strategy_name = filepath.stem
    
    # Get strategy class
    strategy_class = get_strategy_class_name(filepath)
    if not strategy_class:
        return False, f"No Strategy class found"
    
    # Get __main__ block range
    start_line, end_line = get_main_block_range(content)
    if start_line is None:
        return False, f"No __main__ block found"
    
    # Get optimize params
    optimize_params = get_optimize_params(content)
    
    # Get data generation call
    data_gen = get_data_generation_call(content)
    
    # Check for preprocess function
    has_preprocess = 'preprocess' in content.lower()
    
    # Create new __main__ block
    new_main = create_new_main_block(
        strategy_class, strategy_name, optimize_params, data_gen, has_preprocess
    )
    
    # Replace content
    lines = content.split('\n')
    new_lines = lines[:start_line] + [new_main]
    new_content = '\n'.join(new_lines)
    
    if dry_run:
        print(f"  📝 {strategy_name}")
        print(f"     Class: {strategy_class}")
        print(f"     Optimize params: {len(optimize_params or '')} chars")
        return True, "Would update"
    else:
        filepath.write_text(new_content)
        return True, "Updated"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--strategy', type=str)
    args = parser.parse_args()
    
    # Skip already updated
    skip = []  # Empty - we reverted all changes
    
    strategies = sorted(STRATEGIES_DIR.glob('*.py'))
    strategies = [s for s in strategies if not s.stem.startswith('__')]
    
    if args.strategy:
        strategies = [s for s in strategies if args.strategy.lower() in s.stem.lower()]
    
    print(f"{'Applying' if args.apply else 'Dry run'}: {len(strategies)} strategies")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for strategy in strategies:
        if strategy.stem in skip:
            print(f"  ⏭️ Skip: {strategy.stem}")
            continue
        
        success, msg = update_strategy(strategy, dry_run=not args.apply)
        if success:
            if args.apply:
                print(f"  ✅ {strategy.stem}")
            success_count += 1
        else:
            print(f"  ❌ {strategy.stem}: {msg}")
            fail_count += 1
    
    print("=" * 60)
    print(f"Success: {success_count}, Failed: {fail_count}")
    
    if not args.apply:
        print("\nRun with --apply to update files")


if __name__ == '__main__':
    main()
