#!/usr/bin/env python3
"""
Campaign Progress Monitor
=========================
Quick status check for the autonomous improvement campaign.
"""

import json
from pathlib import Path
from datetime import datetime

def check_progress():
    """Display current campaign progress."""
    
    print("="*80)
    print("🌙 AUTONOMOUS CAMPAIGN - PROGRESS CHECK")
    print("="*80)
    print(f"Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if campaign is running
    import subprocess
    result = subprocess.run(
        ["pgrep", "-f", "autonomous_improvement_campaign.py"],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        print("✅ Campaign Status: RUNNING")
        print(f"   PID: {result.stdout.strip()}\n")
    else:
        print("⚠️  Campaign Status: NOT RUNNING (may have completed or crashed)\n")
    
    # Load progress
    progress_file = Path("results/campaign_progress.json")
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        
        print(f"📊 Progress Summary:")
        print(f"   Started: {progress.get('started_at', 'N/A')}")
        print(f"   Last Updated: {progress.get('last_updated', 'N/A')}")
        print(f"   Current Strategy: {progress.get('current_strategy', 'N/A')}")
        print(f"   Current Iteration: {progress.get('current_iteration', 0)}/8")
        print(f"   Strategies Completed: {len(progress.get('strategies_completed', []))}")
        print(f"   Total Backtests: {progress.get('total_backtests', 0)}")
        print(f"   Errors: {progress.get('errors', 0)}\n")
        
        if progress.get('strategies_completed'):
            print("✅ Completed Strategies:")
            for s in progress['strategies_completed']:
                print(f"   - {s['name']}: v{s['best_iteration']} (score: {s['best_score']}/10)")
            print()
    else:
        print("⚠️  No progress file found yet\n")
    
    # Show recent log entries
    log_file = Path("results/campaign_log.txt")
    if log_file.exists():
        print("📝 Recent Log Entries (last 10 lines):")
        print("-"*80)
        with open(log_file) as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"   {line.rstrip()}")
    else:
        print("⚠️  No log file found yet\n")
    
    print("="*80)
    print("\n💡 Commands:")
    print("   - View full log: tail -f results/campaign_log.txt")
    print("   - View progress: cat results/campaign_progress.json | python3 -m json.tool")
    print("   - View errors: cat results/campaign_errors.json | python3 -m json.tool")
    print("   - Stop campaign: pkill -f autonomous_improvement_campaign.py")
    print("="*80)

if __name__ == "__main__":
    check_progress()
