import sys
import os
from pathlib import Path
import pandas as pd
from backtesting import Backtest

# Add root to python path
sys.path.append(str(Path(__file__).parent))

from strategies.vumanchu_cipher_b_deep import VuManchuCipherBDeep

def run_comprehensive_validation():
    # Asset Categories
    categories = {
        "CRYPTO": [
            "data/BTCUSD_15m.csv", 
            "data/ETHUSD_15m.csv", 
            "data/SOLUSD_15m.csv", 
            "data/DOGEUSD_15m.csv",
            "data/XRPUSD_15m.csv"
        ],
        "EQUITIES (Tech/Meme)": [
            "data/NVDA_15m.csv", 
            "data/TSLA_15m.csv", 
            "data/AAPL_15m.csv", 
            "data/GME_15m.csv"
        ],
        "ETFS (Indices)": [
            "data/SPY_15m.csv", 
            "data/QQQ_15m.csv",
        ],
        "HEDGES (Metals/Cmdty)": [
            "data/GLD_15m.csv", 
            "data/SLV_15m.csv", 
            "data/USO_15m.csv"
        ]
    }
    
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE VALIDATION: CYCLE 6 (Deep Pullback)")
    print(f"{'='*60}\n")
    
    for category, file_list in categories.items():
        print(f"\n--- {category} ---")
        cat_results = []
        
        for data_path in file_list:
            if not os.path.exists(data_path):
                # print(f"Skipping {data_path} (Not Found)")
                continue
                
            asset_name = Path(data_path).stem.replace("_15m", "")
            print(f"Testing {asset_name}...", end=" ")
            
            try:
                # Load & Clean Data
                data = pd.read_csv(data_path, parse_dates=True, index_col=0)
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)
                
                # Standardize columns
                # yfinance specific: usually saved as lowercase 'open' by my script
                # Strategy needs 'Open'
                data.columns = data.columns.str.strip()
                # Rename if lowercase
                rename_map = {c: c.capitalize() for c in data.columns if c.islower()}
                if rename_map:
                    data = data.rename(columns=rename_map)
                
                # Filter Required
                req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not set(req_cols).issubset(data.columns):
                    print(f"FAILED (Missing Columns: {set(req_cols) - set(data.columns)})")
                    continue
                
                data = data[req_cols].dropna()
                
                if len(data) < 200:
                    print(f"SKIPPED (Insufficient Data: {len(data)})")
                    continue
                
                # Run Backtest
                bt = Backtest(data, VuManchuCipherBDeep, cash=1_000_000, commission=.002) # 0.2% comm
                stats = bt.run()
                
                res = {
                    "Category": category,
                    "Asset": asset_name,
                    "Return": stats["Return [%]"],
                    "Sharpe": stats["Sharpe Ratio"],
                    "Trades": stats["# Trades"],
                    "Win Rate": stats["Win Rate [%]"],
                    "Factor": stats["Profit Factor"],
                    "DD": stats["Max. Drawdown [%]"]
                }
                cat_results.append(res)
                all_results.append(res)
                print(f"DONE (Ret: {res['Return']:.2f}%)")
                
            except Exception as e:
                print(f"ERROR: {str(e)}")

    # Final Report
    if all_results:
        df = pd.DataFrame(all_results)
        
        print(f"\n{'='*80}")
        print("FINAL PORTFOLIO SUMMARY")
        print(f"{'='*80}")
        print(df.to_string(index=False, float_format="%.2f"))
        print(f"{'-'*80}")
        
        # Aggregate by Category
        print("\nCATEGORY PERFORMANCE:")
        agg = df.groupby("Category").agg({
            "Return": "mean",
            "Sharpe": "mean",
            "Trades": "sum",
            "Win Rate": "mean",
            "Factor": "mean"
        })
        print(agg.to_string(float_format="%.2f"))
        
        # Total Portfolio
        total_ret = df["Return"].mean()
        total_trades = df["Trades"].sum()
        total_wr = df["Win Rate"].mean()
        
        print(f"\nTOTAL PORTFOLIO (Equal Weight):")
        print(f"Avg Return: {total_ret:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Avg Win Rate: {total_wr:.2f}%")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_comprehensive_validation()
