#!/usr/bin/env python3
"""
Close All Conflicting PRs - Closes PRs that can't be merged due to conflicts
Run this after resolve_conflicts_locally.py to clean up remaining PRs
"""

import os
import sys
from dotenv import load_dotenv
from github import Github

load_dotenv()

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO')

def main():
    print("🗑️  Close All Conflicting PRs")
    print("=" * 60)
    
    github = Github(GITHUB_TOKEN)
    repo = github.get_repo(GITHUB_REPO)
    
    # Get all open PRs from Jules
    prs = list(repo.get_pulls(state='open'))
    jules_prs = [pr for pr in prs if pr.head.ref.startswith(('feat/', 'feature/'))]
    
    print(f"\n📊 Found {len(jules_prs)} open Jules PRs")
    
    if not jules_prs:
        print("✅ No PRs to close!")
        return
    
    # Categorize PRs
    dirty_prs = []
    mergeable_prs = []
    
    print("\n🔍 Checking PR statuses...")
    for i, pr in enumerate(jules_prs):
        pr = repo.get_pull(pr.number)
        if pr.mergeable_state in ['dirty', 'blocked']:
            dirty_prs.append(pr)
        elif pr.mergeable_state == 'clean':
            mergeable_prs.append(pr)
        else:
            dirty_prs.append(pr)  # Close anything that isn't clean
        
        if (i + 1) % 20 == 0:
            print(f"   Checked {i+1}/{len(jules_prs)} PRs...")
    
    print(f"\n📈 Status:")
    print(f"   ✅ Mergeable (will keep): {len(mergeable_prs)}")
    print(f"   ❌ Dirty/Blocked (will close): {len(dirty_prs)}")
    
    if not dirty_prs:
        print("\n✅ No conflicting PRs to close!")
        return
    
    print(f"\n⚠️  This will CLOSE {len(dirty_prs)} PRs with conflicts.")
    print("   Jules can recreate them fresh if needed.")
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Aborted")
        return
    
    closed = 0
    failed = []
    
    for i, pr in enumerate(dirty_prs, 1):
        try:
            print(f"[{i}/{len(dirty_prs)}] Closing PR #{pr.number}: {pr.title[:50]}...")
            
            # Add comment explaining why
            pr.create_issue_comment(
                """🗑️ **Closed due to merge conflicts**

This PR has unresolvable merge conflicts with the main branch.
To implement this strategy, please recreate the PR from a fresh branch.

_Closed automatically by cleanup script._
""")
            
            # Close the PR
            pr.edit(state='closed')
            closed += 1
            print(f"   ✅ Closed")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed.append(pr.number)
    
    print("\n" + "=" * 60)
    print("📈 Summary")
    print("=" * 60)
    print(f"Closed: {closed}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\n⚠️  Failed PRs: {failed}")
    
    print(f"\n💡 Remaining mergeable PRs: {len(mergeable_prs)}")
    if mergeable_prs:
        print("   Run 'python manual_merge_prs.py' to merge them.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
