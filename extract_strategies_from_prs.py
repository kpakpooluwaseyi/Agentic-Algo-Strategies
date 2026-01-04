#!/usr/bin/env python3
"""
🔧 Extract strategies from conflicting PRs and add them directly to main.
Since all PRs conflict on results/temp_result.json, we extract just the strategy files.
"""

import os
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv
from github import Github, Auth

load_dotenv()

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO', 'kpakpooluwaseyi/moon-dev-ai-agents-for-trading')

def main():
    gh = Github(auth=Auth.Token(GITHUB_TOKEN))
    repo = gh.get_repo(GITHUB_REPO)
    
    # Get approved PRs
    prs = list(repo.get_pulls(state='open'))
    ready = [p for p in prs if not p.draft]
    
    approved_prs = []
    for pr in ready:
        comments = list(pr.get_issue_comments())
        if any('APPROVED' in c.body and 'PR Gatekeeper' in c.body for c in comments):
            approved_prs.append(pr)
    
    print(f"📦 Found {len(approved_prs)} approved PRs with conflicts")
    
    # For each PR, get the strategy file content directly
    strategies_added = 0
    
    for pr in approved_prs:
        try:
            files = list(pr.get_files())
            strategy_file = None
            
            for f in files:
                if f.filename.startswith('strategies/') and f.filename.endswith('.py'):
                    strategy_file = f
                    break
            
            if not strategy_file:
                print(f"⚠️ PR #{pr.number}: No strategy file found")
                continue
            
            # Get the file content from the PR branch
            content = repo.get_contents(strategy_file.filename, ref=pr.head.ref)
            strategy_code = content.decoded_content.decode('utf-8')
            
            # Check if file already exists in main
            try:
                existing = repo.get_contents(strategy_file.filename, ref='main')
                print(f"⏭️ PR #{pr.number}: {strategy_file.filename} already exists in main")
                # Close the PR since strategy is already there
                pr.create_issue_comment("🤖 Strategy file already exists in main. Closing as duplicate.")
                pr.edit(state='closed')
                continue
            except:
                pass  # File doesn't exist, we'll create it
            
            # Create the file in main
            repo.create_file(
                path=strategy_file.filename,
                message=f"🤖 Add strategy from PR #{pr.number}: {pr.title[:50]}",
                content=strategy_code,
                branch='main'
            )
            
            strategies_added += 1
            print(f"✅ PR #{pr.number}: Added {strategy_file.filename}")
            
            # Close the PR
            pr.create_issue_comment("🤖 Strategy extracted and added to main. Closing PR.")
            pr.edit(state='closed')
            
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ PR #{pr.number}: {str(e)[:80]}")
    
    print(f"\n📊 Summary: Added {strategies_added} strategies to main")
    print(f"🔄 Run 'git pull' to get the new files locally")

if __name__ == '__main__':
    main()
