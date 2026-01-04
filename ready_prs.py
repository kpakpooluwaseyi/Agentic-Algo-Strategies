import os
import time
from github import Github
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "kpakpooluwaseyi/Agentic-Algo-Strategies")

def ready_the_prs():
    if not GITHUB_TOKEN:
        print("❌ GITHUB_TOKEN not found!")
        return

    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(GITHUB_REPO)
    
    print(f"Connecting to {GITHUB_REPO}...")
    prs = list(repo.get_pulls(state='open'))
    
    jules_drafts = []
    for pr in prs:
        # Check if it's a Jules PR and in draft
        if pr.draft and pr.user.login == "google-labs-jules[bot]":
            jules_drafts.append(pr)
            
    if not jules_drafts:
        print("📭 No Jules draft PRs found.")
        return

    print(f"🚀 Found {len(jules_drafts)} draft PRs from Jules. Readying them now...")
    
    for pr in jules_drafts:
        try:
            # PyGithub doesn't have a direct 'ready_for_review' method in older versions
            # But we can use the GraphQL API or just update the via patch if supported
            # Actually, we can use the REST API via a custom request if needed, 
            # but let's try the direct state update first.
            
            # GraphQL is the most reliable way to change draft status
            query = """
            mutation {
              markPullRequestReadyForReview(input: {pullRequestId: "%s"}) {
                pullRequest {
                  id
                  isDraft
                }
              }
            }
            """ % pr.node_id
            
            gh._Github__requester.requestJsonAndCheck(
                "POST", 
                "/graphql", 
                input={"query": query}
            )
            print(f"✅ PR #{pr.number} is now READY")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Failed to ready PR #{pr.number}: {e}")

if __name__ == "__main__":
    ready_the_prs()
