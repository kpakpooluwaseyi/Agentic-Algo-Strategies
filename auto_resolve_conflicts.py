
import os
import time
import subprocess
from github import Github
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO', 'kpakpooluwaseyi/moon-dev-ai-agents-for-trading')

def run_command(command):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            command,
            cwd=os.getcwd(),
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{command}': {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None

def resolve_conflicts_for_pr(pr, github_token):
    """
    Resolves conflicts for a single PR by merging main and favoring main's config.
    """
    print(f"\n🔧 Processing PR #{pr.number}: {pr.title}")
    print(f"   Branch: {pr.head.ref}")

    # 1. Fetch the PR branch
    print("   Fetching branch...")
    run_command(f"git fetch origin {pr.head.ref}:{pr.head.ref}")
    
    # 2. Checkout the branch
    run_command(f"git checkout {pr.head.ref}")
    
    # 3. Attempt to merge main
    print("   Merging main into branch...")
    # We expect this to fail if there are conflicts
    try:
        subprocess.run(
            "git merge origin/main --no-edit",
            cwd=os.getcwd(),
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print("   ✅ Merged cleanly without conflicts.")
        # Push anyway to update branch
        push_cmd = f"git push origin {pr.head.ref}"
        run_command(push_cmd)
        return True
        
    except subprocess.CalledProcessError:
        print("   ⚠️  Merge conflict detected. Resolving...")
    
    # 4. Resolve Conflicts
    
    # A. requirements.txt -> Take Main (theirs)
    if os.path.exists("requirements.txt"):
        print("   Resolving requirements.txt (taking main)...")
        run_command("git checkout --theirs requirements.txt")
        run_command("git add requirements.txt")

    # B. results/temp_result.json -> Remove or Take Main (which shouldn't have it)
    # Actually if main removed it, and PR has it modified, it might be a modify/delete conflict
    # Safest is to remove it from index
    print("   Removing temporary result files...")
    run_command("git rm --cached results/temp_result.json 2>/dev/null")
    run_command("git rm results/temp_result.json 2>/dev/null")
    run_command("git rm --cached results/*.html 2>/dev/null")
    run_command("git rm results/*.html 2>/dev/null")
    # Also ignore leaderboard if conflicted (take main's)
    if os.path.exists("results/leaderboard.csv"):
        run_command("git checkout --theirs results/leaderboard.csv 2>/dev/null")
        run_command("git add results/leaderboard.csv 2>/dev/null")

    # C. Commit resolution
    print("   Committing resolution...")
    commit_msg = "fix: resolve merge conflicts with main"
    run_command(f'git commit -m "{commit_msg}"')
    
    # 5. Push changes
    print("   Pushing changes...")
    # Use the token for auth if needed, but assuming local git is configured
    # python-github doesn't help with git push, we rely on local git credentials
    push_result = run_command(f"git push origin {pr.head.ref}")
    
    if push_result is not None:
        print("   ✅ Successfully resolved and pushed.")
        return True
    else:
        print("   ❌ Failed to push changes.")
        return False

def main():
    print("🚀 Auto Conflict Resolver Started")
    
    if not GITHUB_TOKEN:
        print("❌ Error: GITHUB_TOKEN not found")
        return

    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO)
    
    # Get all open PRs
    # We only care about Jules PRs basically
    prs = repo.get_pulls(state='open', sort='created', direction='desc')
    
    total_processed = 0
    success_count = 0
    
    # Limit to batches to avoid overwhelming
    BATCH_SIZE = 10
    
    print("🔎 Scanning for unmergeable PRs...")
    
    conflicted_prs = []
    for pr in prs:
        if pr.mergeable is False: # specific check for False, not None (None means unknown)
            # check if it's a Jules PR
            if pr.head.ref.startswith('feat/') or pr.head.ref.startswith('feature/'):
                conflicted_prs.append(pr)
    
    print(f"Found ~{len(conflicted_prs)} conflicted PRs. Processing first {BATCH_SIZE}...")
    
    # Clean up local git state first
    run_command("git checkout main")
    run_command("git pull origin main")
    
    for pr in conflicted_prs[:BATCH_SIZE]:
        if resolve_conflicts_for_pr(pr, GITHUB_TOKEN):
            success_count += 1
        total_processed += 1
        time.sleep(2) # be nice to API
        
        # Switch back to main between PRs to be safe
        run_command("git checkout main")
        
    print(f"\n✨ Batch Complete. Resolved {success_count}/{total_processed} PRs.")

if __name__ == "__main__":
    main()
