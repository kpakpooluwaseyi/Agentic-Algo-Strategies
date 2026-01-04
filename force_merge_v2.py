import os
import sys
import time
import subprocess
from github import Github
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO', 'kpakpooluwaseyi/Agentic-Algo-Strategies')

def run_git(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Git error: {result.stderr}")
    return result

def force_merge_v2():
    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(GITHUB_REPO)
    
    # Get recent PRs
    prs = list(repo.get_pulls(state='open', sort='created', direction='desc'))
    v2_prs = [pr for pr in prs if pr.number >= 670]
    
    if not v2_prs:
        print("📭 No recent V2 PRs found.")
        return

    print(f"🔧 Starting Force Merge for {len(v2_prs)} V2 strategies...")
    
    # Prep local repo
    run_git("git checkout main")
    run_git("git fetch origin main")
    run_git("git reset --hard origin/main")

    for pr in v2_prs:
        branch = pr.head.ref
        print(f"\n--- PR #{pr.number}: {pr.title} ({branch}) ---")
        
        try:
            # 1. Fetch and checkout
            run_git(f"git fetch origin {branch}:{branch} --force")
            run_git(f"git checkout {branch}")
            
            # 2. Merge main into branch
            merge = run_git("git merge main --no-edit")
            if "CONFLICT" in merge.stdout or merge.returncode != 0:
                print("⚠️  Conflicts detected. Auto-resolving...")
                # Keep our strategies, but take main's version of the shared temp_result.json (doesn't matter)
                run_git("git checkout --ours strategies/")
                run_git("git checkout --theirs results/temp_result.json")
                run_git("git add .")
                run_git('git commit -m "Auto-resolve transient conflicts for V2 merge"')
            
            # 3. Push back
            print("📤 Pushing resolved branch...")
            push = run_git(f"git push origin {branch} --force")
            if push.returncode != 0:
                print(f"❌ Failed to push {branch}")
                continue
                
            # 4. Merge on GitHub
            time.sleep(2)
            pr = repo.get_pull(pr.number)
            if pr.mergeable:
                pr.merge(commit_message=f"🤖 Auto-merge V2 Strategy: {pr.title}")
                print(f"✅ PR #{pr.number} MERGED!")
            else:
                print(f"❌ PR #{pr.number} still not mergeable (state: {pr.mergeable_state})")
                
        except Exception as e:
            print(f"❌ Error processing PR #{pr.number}: {e}")
            
    # Back to main
    run_git("git checkout main")

if __name__ == "__main__":
    force_merge_v2()
