import csv
from pathlib import Path

FILES = [
    Path('/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/results/leaderboard.csv'),
    Path('/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/results/leaderboard_merged.csv')
]

# Target columns (15 total)
NEW_HEADER = [
    'timestamp', 'strategy_name', 'dataset_name', 
    'return_pct', 'sharpe_ratio', 'max_drawdown_pct', 
    'win_rate_pct', 'total_trades', 'status',
    'wfa_status', 'wfa_oos_return_pct', 'wfa_degradation_pct',
    'wfo_status', 'wfo_oos_return_pct', 'wfo_degradation_pct'
]

def repair_csv(file_path):
    if not file_path.exists():
        print(f"File {file_path} not found. Skipping.")
        return

    new_file = file_path.with_suffix('.csv.new')
    with open(file_path, 'r', newline='') as f_in, open(new_file, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        # Write new header
        writer.writerow(NEW_HEADER)
        
        # Skip original header
        original_header = next(reader)
        print(f"Processing {file_path.name}. Original header: {original_header}")
        
        # Map indices based on original header to be more robust
        h_map = {col: i for i, col in enumerate(original_header)}
        
        rows_repaired = 0
        for row in reader:
            if not row: continue
            
            # Create a 15-column row with default values
            new_row = ['' for _ in range(15)]
            
            # Standard mappings (present in most formats)
            def get_val(col_name, default=None):
                if col_name in h_map and h_map[col_name] < len(row):
                    return row[h_map[col_name]]
                return default

            new_row[0] = get_val('timestamp')
            new_row[1] = get_val('strategy_name')
            new_row[2] = get_val('dataset_name', 'unknown')
            new_row[3] = get_val('return_pct')
            new_row[4] = get_val('sharpe_ratio')
            new_row[5] = get_val('max_drawdown_pct')
            new_row[6] = get_val('win_rate_pct')
            new_row[7] = get_val('total_trades')
            new_row[8] = get_val('status', 'SUCCESS')
            new_row[9] = get_val('wfa_status', 'SKIPPED')
            new_row[10] = get_val('wfa_oos_return_pct')
            new_row[11] = get_val('wfa_degradation_pct')
            new_row[12] = get_val('wfo_status', 'SKIPPED')
            new_row[13] = get_val('wfo_oos_return_pct')
            new_row[14] = get_val('wfo_degradation_pct')
            
            writer.writerow(new_row)
            rows_repaired += 1
            
    # Swap files
    file_path.with_suffix('.bak').unlink(missing_ok=True)
    file_path.rename(file_path.with_suffix('.bak'))
    new_file.rename(file_path)
    print(f"Successfully repaired {rows_repaired} rows in {file_path.name}.")

if __name__ == '__main__':
    for f in FILES:
        repair_csv(f)

