#!/usr/bin/env python3
"""
Draft PR Converter & Merger - Mark Jules draft PRs as ready and merge them
"""

import os
import sys
from dotenv import load_dotenv
from github import Github

load_dotenv()

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO')

def main():
    print("🔧 Draft PR Converter & Merger for Jules Strategies")
    print("=" * 60)
    
    github = Github(GITHUB_TOKEN)
    repo = github.get_repo(GITHUB_REPO)
    
    # Get all open PRs
    prs = list(repo.get_pulls(state='open'))
    jules_prs = [pr for pr in prs if pr.head.ref.startswith(('feat/', 'feature/'))]
    draft_prs = [pr for pr in jules_prs if pr.draft]
    
    print(f"\n📊 Statistics:")
    print(f"   Total Jules PRs: {len(jules_prs)}")
    print(f"   Draft PRs: {len(draft_prs)}")
    print(f"   Ready PRs: {len(jules_prs) - len(draft_prs)}")
    
    if len(jules_prs) == 0:
        print("\n✅ No Jules PRs to process!")
        return
    
    # Ask for confirmation
    print(f"\n⚠️  This will:")
    if draft_prs:
        print(f"   1. Mark {len(draft_prs)} draft PRs as ready for review")
    print(f"   2. Merge all {len(jules_prs)} PRs from Jules")
    
    response = input("\nProceed? (yes/no): ")
    
    if response.lower() != 'yes':
        print("❌ Aborted by user")
        return
    
    # Step 1: Convert drafts to ready
    if draft_prs:
        print(f"\n📝 Step 1: Converting draft PRs to ready...")
        converted = 0
        failed_convert = []
        
        for i, pr in enumerate(draft_prs, 1):
            try:
                print(f"[{i}/{len(draft_prs)}] PR #{pr.number}: {pr.title[:60]}...")
                
                # Mark as ready for review using GraphQL mutation
                mutation = """
                mutation($pullRequestId: ID!) {
                  markPullRequestReadyForReview(input: {pullRequestId: $pullRequestId}) {
                    pullRequest {
                      id
                      number
                    }
                  }
                }
                """
                
                # Get PR node ID for GraphQL
                pr_node_id = pr.raw_data['node_id']
                
                # Execute GraphQL mutation
                github._Github__requester.requestJsonAndCheck(
                    "POST",
                    "/graphql",
                    input={
                        "query": mutation,
                        "variables": {"pullRequestId": pr_node_id}
                    }
                )
                
                converted += 1
                print(f"   ✅ Marked as ready")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                failed_convert.append(pr.number)
        
        print(f"\n✅ Converted {converted}/{len(draft_prs)} draft PRs to ready")
    
    # Step 2: Merge all PRs
    print(f"\n🔀 Step 2: Merging PRs...")
    merged = 0
    failed_merge = []
    
    # Refresh PR list to get updated draft status
    jules_prs = [pr for pr in repo.get_pulls(state='open') if pr.head.ref.startswith(('feat/', 'feature/'))]
    
    batch_size = 10
    for i, pr in enumerate(jules_prs, 1):
        try:
            # Refresh PR object to get latest state
            pr = repo.get_pull(pr.number)
            
            if pr.draft:
                print(f"[{i}/{len(jules_prs)}] PR #{pr.number}: Still draft, skipping")
                failed_merge.append(pr.number)
                continue
            
            # Wait for GitHub to compute mergeable state (can be None initially)
            import time
            for retry in range(5):
                if pr.mergeable is not None:
                    break
                time.sleep(2)
                pr = repo.get_pull(pr.number)
            
            if pr.mergeable is False or pr.mergeable_state == 'dirty':
                print(f"[{i}/{len(jules_prs)}] PR #{pr.number}: Not mergeable (state: {pr.mergeable_state}), skipping")
                failed_merge.append(pr.number)
                continue
            
            print(f"[{i}/{len(jules_prs)}] Merging PR #{pr.number}: {pr.title[:60]}...")
            
            # Add approval comment
            pr.create_issue_comment(
                """✅ **Batch Merge - APPROVED**
                
**Status:** Marked ready and batch merged
**Reason:** Jules-generated strategy

Strategy will be executed by Local Runner after merge.
""")
            
            # Merge the PR
            pr.merge(commit_message=f"🤖 Batch merge: Jules strategy #{pr.number}")
            
            merged += 1
            print(f"   ✅ Merged successfully")
            
            # Pause after each batch
            if i % batch_size == 0 and i < len(jules_prs):
                print(f"\n⏸️  Completed batch of {batch_size}. Pausing...")
                input("Press Enter to continue with next batch...")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed_merge.append(pr.number)
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 Final Summary")
    print("=" * 60)
    if draft_prs:
        print(f"Draft PRs converted: {converted}/{len(draft_prs)}")
    print(f"PRs merged: {merged}/{len(jules_prs)}")
    
    if failed_merge:
        print(f"\n⚠️  Failed to merge: {failed_merge[:20]}")
        if len(failed_merge) > 20:
            print(f"   ... and {len(failed_merge) - 20} more")
    
    if merged > 0:
        print(f"\n🎉 Success! {merged} strategies merged!")
        print(f"\n💡 Next steps:")
        print(f"   1. Local Runner will pull and execute strategies")
        print(f"   2. Results will appear in results/leaderboard.csv")
        print(f"   3. Monitor: tail -f logs/local_runner.log")

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
