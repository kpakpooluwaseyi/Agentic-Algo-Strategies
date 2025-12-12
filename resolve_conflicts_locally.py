#!/usr/bin/env python3
"""
Local Conflict Resolver - Fetches PR branches, resolves conflicts locally, pushes back
This handles cases where GitHub's update_branch API can't auto-resolve.
"""

import os
import sys
import subprocess
from dotenv import load_dotenv
from github import Github

load_dotenv()

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO')

def run_git(cmd, check=True, capture=True):
    """Run a git command"""
    result = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True
    )
    if check and result.returncode != 0:
        raise Exception(f"Git command failed: {cmd}\n{result.stderr}")
    return result

def main():
    print("🔧 Local Conflict Resolver")
    print("=" * 60)
    print("This script resolves conflicts by merging main into each PR branch locally.")
    print("For strategy files, we'll keep BOTH versions (since they're separate strategies).")
    print("")
    
    github = Github(GITHUB_TOKEN)
    repo = github.get_repo(GITHUB_REPO)
    
    # Ensure we're on main and up to date
    print("📥 Updating main branch...")
    run_git("git checkout main")
    run_git("git fetch origin main")
    run_git("git reset --hard origin/main")  # Force sync with origin
    
    # Get open PRs with conflicts
    prs = list(repo.get_pulls(state='open'))
    jules_prs = [pr for pr in prs if pr.head.ref.startswith(('feat/', 'feature/'))]
    
    # Check for dirty PRs
    dirty_prs = []
    print(f"\n🔍 Checking {len(jules_prs)} Jules PRs...")
    for pr in jules_prs:
        pr = repo.get_pull(pr.number)
        if pr.mergeable_state == 'dirty':
            dirty_prs.append(pr)
    
    print(f"\n📊 Found {len(dirty_prs)} PRs with conflicts")
    
    if not dirty_prs:
        print("✅ No dirty PRs to fix!")
        return
    
    print(f"\n⚠️  This will:")
    print(f"   1. Checkout each PR branch locally")
    print(f"   2. Merge main into it (accepting theirs for strategies)")
    print(f"   3. Push the resolved branch back")
    print(f"   4. Merge the PR on GitHub")
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Aborted")
        return
    
    resolved = 0
    merged = 0
    failed = []
    
    for i, pr in enumerate(dirty_prs, 1):
        branch = pr.head.ref
        print(f"\n[{i}/{len(dirty_prs)}] PR #{pr.number}: {branch}")
        
        try:
            # Fetch the branch
            print(f"   📥 Fetching branch...")
            run_git(f"git fetch origin {branch}:{branch} --force 2>/dev/null || git fetch origin {branch}")
            
            # Checkout the branch  
            print(f"   🔀 Checking out branch...")
            run_git(f"git checkout {branch}")
            
            # Try to merge main
            print(f"   🔄 Merging main...")
            merge_result = run_git("git merge main --no-edit", check=False)
            
            if merge_result.returncode != 0:
                # There are conflicts - auto-resolve strategy files
                print(f"   ⚠️  Conflicts detected, auto-resolving...")
                
                # For strategy files in strategies/, keep the PR's version (ours)
                run_git("git checkout --ours strategies/ 2>/dev/null || true", check=False)
                
                # For other files like leaderboard, take theirs (main's)
                run_git("git checkout --theirs results/ 2>/dev/null || true", check=False)
                
                # Stage all resolved files
                run_git("git add -A")
                
                # Check if there are still unresolved conflicts
                status = run_git("git status --porcelain", check=False)
                if "UU " in status.stdout or "AA " in status.stdout:
                    print(f"   ❌ Unresolved conflicts remain")
                    run_git("git merge --abort", check=False)
                    failed.append(pr.number)
                    continue
                
                # Commit the merge
                run_git('git commit -m "Merge main into feature branch (auto-resolved)"')
                resolved += 1
                print(f"   ✅ Conflicts resolved")
            else:
                resolved += 1
                print(f"   ✅ Merged cleanly")
            
            # Push back to GitHub
            print(f"   📤 Pushing to GitHub...")
            run_git(f"git push origin {branch}")
            
            # Go back to main
            run_git("git checkout main")
            
            # Now try to merge the PR on GitHub
            print(f"   🔀 Merging PR on GitHub...")
            import time
            time.sleep(2)  # Give GitHub time to update
            
            pr = repo.get_pull(pr.number)  # Refresh
            if pr.mergeable:
                pr.merge(commit_message=f"🤖 Auto-merge: {branch}")
                merged += 1
                print(f"   ✅ PR merged!")
            else:
                print(f"   ⏳ PR will be mergeable soon (state: {pr.mergeable_state})")
                # It might take a moment for GitHub to compute mergeable state
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed.append(pr.number)
            # Make sure we're on main
            run_git("git checkout main", check=False)
            run_git("git merge --abort", check=False)
    
    # Clean up - go back to main
    run_git("git checkout main", check=False)
    
    print("\n" + "=" * 60)
    print("📈 Summary")
    print("=" * 60)
    print(f"Resolved conflicts: {resolved}")
    print(f"Merged on GitHub: {merged}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\n⚠️  Failed PRs: {failed}")
    
    if merged < resolved:
        print(f"\n💡 Some PRs were resolved but not merged yet.")
        print(f"   Run 'python manual_merge_prs.py' to merge them.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Interrupted")
        subprocess.run("git checkout main", shell=True, capture_output=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        subprocess.run("git checkout main", shell=True, capture_output=True)
        sys.exit(1)
