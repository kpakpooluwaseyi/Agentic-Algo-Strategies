import pandas as pd
from pathlib import Path

FILE = Path('/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/results/leaderboard.csv')

if FILE.exists():
    df = pd.read_csv(FILE)
    cols_added = False
    for col in ['wfo_status', 'wfo_oos_return_pct', 'wfo_degradation_pct']:
        if col not in df.columns:
            df[col] = 'SKIPPED' if 'status' in col else None
            cols_added = True
    
    if cols_added:
        df.to_csv(FILE, index=False)
        print(f"Updated {FILE.name} schema.")
    else:
        print(f"{FILE.name} already up to date.")
else:
    print(f"{FILE.name} not found.")
