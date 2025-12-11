yrse#!/usr/bin/env python3
"""
Force Merge PRs - Handles conflicts by rebasing or resetting
Uses GitHub's update branch feature + merge
"""

import os
import sys
import time
from dotenv import load_dotenv
from github import Github, GithubException

load_dotenv()

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO')

def main():
    print("🔧 Force Merge PRs - Handles Conflicts")
    print("=" * 60)
    
    github = Github(GITHUB_TOKEN)
    repo = github.get_repo(GITHUB_REPO)
    
    # Get all open PRs from Jules
    prs = list(repo.get_pulls(state='open'))
    jules_prs = [pr for pr in prs if pr.head.ref.startswith(('feat/', 'feature/'))]
    
    print(f"\n📊 Found {len(jules_prs)} Jules PRs")
    
    # Categorize PRs
    mergeable = []
    conflicts = []
    drafts = []
    unknown = []
    
    print("\n🔍 Checking PR status (this may take a moment)...")
    
    for i, pr in enumerate(jules_prs):
        # Refresh to get mergeable status
        pr = repo.get_pull(pr.number)
        
        if pr.draft:
            drafts.append(pr)
        elif pr.mergeable_state == 'clean':
            mergeable.append(pr)
        elif pr.mergeable_state in ['dirty', 'behind']:
            conflicts.append(pr)
        else:
            unknown.append(pr)
        
        if (i + 1) % 20 == 0:
            print(f"   Checked {i+1}/{len(jules_prs)} PRs...")
    
    print(f"\n📈 Status:")
    print(f"   ✅ Mergeable: {len(mergeable)}")
    print(f"   ⚠️  With conflicts: {len(conflicts)}")
    print(f"   📝 Drafts: {len(drafts)}")
    print(f"   ❓ Unknown: {len(unknown)}")
    
    if not mergeable and not conflicts:
        print("\n✅ No PRs to process!")
        return
    
    print(f"\n⚠️  This will:")
    print(f"   1. Merge {len(mergeable)} clean PRs")
    print(f"   2. Update branch & merge {len(conflicts)} conflict PRs")
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Aborted by user")
        return
    
    merged = 0
    failed = []
    
    # Step 1: Merge clean PRs
    print(f"\n🔀 Step 1: Merging {len(mergeable)} clean PRs...")
    for i, pr in enumerate(mergeable, 1):
        try:
            print(f"[{i}/{len(mergeable)}] PR #{pr.number}: {pr.title[:50]}...")
            pr.merge(commit_message=f"🤖 Auto-merge: Jules strategy #{pr.number}")
            merged += 1
            print(f"   ✅ Merged")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed.append(pr.number)
    
    # Step 2: Handle conflicts by updating branch first
    print(f"\n🔄 Step 2: Updating & merging {len(conflicts)} PRs with conflicts...")
    for i, pr in enumerate(conflicts, 1):
        try:
            print(f"[{i}/{len(conflicts)}] PR #{pr.number}: {pr.title[:50]}...")
            
            # Try to update the branch (rebase with main)
            try:
                pr.update_branch()
                print(f"   ⬆️  Branch updated")
                time.sleep(2)  # Wait for GitHub to process
            except GithubException as e:
                if 'merge conflict' in str(e).lower():
                    print(f"   ⚠️  Cannot auto-resolve conflict")
                    failed.append(pr.number)
                    continue
                else:
                    # update_branch might not be available, try merge anyway
                    pass
            
            # Refresh PR and try merge
            pr = repo.get_pull(pr.number)
            
            if pr.mergeable:
                pr.merge(commit_message=f"🤖 Auto-merge: Jules strategy #{pr.number}")
                merged += 1
                print(f"   ✅ Merged")
            else:
                print(f"   ❌ Still not mergeable (state: {pr.mergeable_state})")
                failed.append(pr.number)
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed.append(pr.number)
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 Final Summary")
    print("=" * 60)
    print(f"Successfully merged: {merged}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\n⚠️  Failed PRs (need manual conflict resolution): {failed[:20]}")
        if len(failed) > 20:
            print(f"   ... and {len(failed) - 20} more")
        
        print(f"\n💡 To manually resolve conflicts:")
        print(f"   1. gh pr checkout <PR_NUMBER>")
        print(f"   2. git merge main")
        print(f"   3. Resolve conflicts and commit")
        print(f"   4. git push")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
