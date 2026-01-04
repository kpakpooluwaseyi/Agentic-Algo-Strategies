#!/usr/bin/env python3
"""
🌙 Moon Dev's PR Gatekeeper
Audits Pull Requests from Jules for safety before merging.

Uses OpenRouter free tier (GPT-OSS-120B or Qwen 3) for code auditing.

Safety checks:
- Malware patterns (subprocess, os.system, eval, exec)
- Infinite loops
- Suspicious imports (socket, urllib for non-data purposes)
- Resource exhaustion patterns
- Data exfiltration attempts

Usage:
    python pr_gatekeeper.py             # Audit and process PRs
    python pr_gatekeeper.py --dry-run   # Preview without merging
    python pr_gatekeeper.py --auto      # Auto-merge safe PRs (use with caution)
"""

import os
import sys
import re
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from dotenv import load_dotenv
from github import Github
import requests

# Load environment variables
load_dotenv()

# Configuration
LOGS_DIR = Path("logs")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "kpakpooluwaseyi/Agentic-Algo-Strategies")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# Free tier models - try these in order
OPENROUTER_MODELS = [
    "qwen/qwen3-235b-a22b:free",  # Qwen 3 free tier
    "google/gemma-3-27b-it:free",  # Gemma 3 free
    "meta-llama/llama-3.3-70b-instruct:free",  # Llama 3.3 free
]

# Setup logging
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "pr_gatekeeper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Safety audit prompt - IMPORTANT: Be lenient for normal trading strategy patterns!
AUDIT_PROMPT = """You are a practical code reviewer for auto-generated trading strategy code.

**CONTEXT**: These are backtesting strategies for algorithmic trading research. They are run in isolated environments, NOT in production. Be practical, not paranoid.

## APPROVE these NORMAL patterns (do NOT reject):
- ✅ Reading CSV/data files from `data/` directory (standard practice)
- ✅ Generating synthetic data for backtesting (numpy.random, monte carlo)
- ✅ Using sklearn, pandas, numpy for ML models and analysis
- ✅ Using backtesting.py, backtrader frameworks
- ✅ Using pandas_ta, talib for indicators
- ✅ Writing results to `results/` directory
- ✅ Loading data with pd.read_csv()
- ✅ Plotting with matplotlib
- ✅ JSON serialization of results
- ✅ Random forest, linear regression, any ML models

## REJECT only ACTUAL security threats:
- ❌ Network requests to unknown external servers (NOT data APIs)
- ❌ eval(), exec(), compile() with user input
- ❌ subprocess.Popen with shell=True and untrusted input
- ❌ Accessing credentials, API keys, .env contents to exfiltrate
- ❌ Writing files outside project directory (path traversal)
- ❌ Deliberate infinite loops (while True without break)
- ❌ socket connections to external servers

## IMPORTANT GUIDELINES:
- Reading from `data/BTC-USD-15m.csv` is NORMAL - APPROVE
- Using sklearn.ensemble.RandomForestRegressor is NORMAL - APPROVE  
- Generating synthetic test data is NORMAL - APPROVE
- These strategies will be tested locally, not deployed to production

---
CODE DIFF TO AUDIT:

{diff_content}

---
Respond with EXACTLY one of these formats:

If SAFE (the code looks like normal trading strategy code):
```
VERDICT: APPROVE
REASON: [Brief explanation]
```

If UNSAFE (you found an actual security threat from the REJECT list above):
```
VERDICT: REJECT
REASON: [Specific security threat found - must be from REJECT list]
CODE_LOCATION: [Exact line or section]
```

**When in doubt, APPROVE.** These are research strategies, not production code.
"""


def init_github():
    """Initialize GitHub client"""
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN not found in environment!")
    
    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(GITHUB_REPO)
    logger.info(f"✅ Connected to GitHub repo: {GITHUB_REPO}")
    return gh, repo


def call_openrouter(prompt: str) -> Optional[str]:
    """Call OpenRouter API with the given prompt"""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment!")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/moon-dev-trading",
        "X-Title": "Moon Dev PR Gatekeeper"
    }
    
    for model in OPENROUTER_MODELS:
        try:
            logger.info(f"🤖 Trying model: {model}")
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.1  # Low temperature for consistent judgments
            }
            
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                logger.info(f"✅ Got response from {model}")
                return content
            else:
                logger.warning(f"⚠️ Model {model} returned {response.status_code}: {response.text[:200]}")
                continue
                
        except Exception as e:
            logger.warning(f"⚠️ Model {model} failed: {e}")
            continue
    
    logger.error("❌ All OpenRouter models failed")
    return None


def get_jules_prs(repo) -> List:
    """Get all open PRs from Jules (auto-generated branches)
    
    Jules branch patterns:
    - feat/*
    - feature/*
    - add-*-strategy-*
    - *-strategy-{large_number}
    - jules-*
    """
    prs = list(repo.get_pulls(state='open'))
    jules_prs = []
    
    for pr in prs:
        branch = pr.head.ref.lower()
        
        # Match various Jules patterns
        is_jules = (
            branch.startswith(('feat/', 'feature/', 'jules-')) or
            'strategy' in branch or
            # Branches with large numeric suffixes are typically auto-generated
            re.search(r'-\d{10,}$', branch) is not None
        )
        
        if is_jules:
            jules_prs.append(pr)
    
    logger.info(f"📋 Found {len(jules_prs)} open Jules PR(s)")
    return jules_prs


def get_pr_diff(pr) -> Optional[str]:
    """Get the diff content for a PR"""
    try:
        files = pr.get_files()
        diff_content = []
        
        for file in files:
            diff_content.append(f"=== {file.filename} ===")
            if file.patch:
                diff_content.append(file.patch)
            else:
                diff_content.append("[Binary or empty file]")
            diff_content.append("")
        
        full_diff = "\n".join(diff_content)
        logger.info(f"📄 Got diff: {len(full_diff)} chars, {len(list(files))} file(s)")
        return full_diff
        
    except Exception as e:
        logger.error(f"❌ Failed to get PR diff: {e}")
        return None


def parse_verdict(response: str) -> Tuple[str, str]:
    """Parse the verdict from OpenRouter response"""
    response = response.strip()
    
    # Look for VERDICT line
    verdict_match = re.search(r'VERDICT:\s*(APPROVE|REJECT)', response, re.IGNORECASE)
    reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
    
    if verdict_match:
        verdict = verdict_match.group(1).upper()
        reason = reason_match.group(1).strip() if reason_match else "No reason provided"
        return verdict, reason
    
    # If no clear verdict, default to REJECT for safety
    logger.warning("⚠️ Could not parse verdict, defaulting to REJECT")
    return "REJECT", "Unable to determine safety - manual review required"


def audit_pr(pr) -> Tuple[str, str]:
    """Audit a single PR and return verdict"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔍 Auditing PR #{pr.number}: {pr.title[:50]}...")
    
    # Get diff
    diff = get_pr_diff(pr)
    if not diff:
        return "REJECT", "Failed to retrieve PR diff"
    
    # Quick pre-check for obvious malware patterns
    dangerous_patterns = [
        (r'subprocess\.(call|run|Popen).*shell\s*=\s*True', "Shell execution with shell=True"),
        (r'eval\s*\(', "Use of eval()"),
        (r'exec\s*\(', "Use of exec()"),
        (r'__import__\s*\(', "Dynamic import"),
        (r'socket\.(socket|connect)', "Raw socket usage"),
        (r'requests\.(get|post).*(?!binance|coingecko|yahoo)', "Suspicious HTTP request"),
    ]
    
    for pattern, description in dangerous_patterns:
        if re.search(pattern, diff, re.IGNORECASE):
            logger.warning(f"🚨 Quick-check found: {description}")
            # Still send to LLM for context-aware analysis
    
    # Send to OpenRouter for detailed analysis
    prompt = AUDIT_PROMPT.format(diff_content=diff[:50000])  # Limit size
    response = call_openrouter(prompt)
    
    if not response:
        return "REJECT", "OpenRouter API unavailable - manual review required"
    
    verdict, reason = parse_verdict(response)
    logger.info(f"📋 Verdict: {verdict}")
    logger.info(f"📝 Reason: {reason[:200]}")
    
    return verdict, reason


def process_pr(repo, pr, verdict: str, reason: str, dry_run: bool = False, auto_merge: bool = False) -> bool:
    """Process a PR based on audit verdict"""
    try:
        if verdict == "APPROVE":
            comment = f"""✅ **Security Audit: APPROVED**

🤖 Automated safety check by Moon Dev's PR Gatekeeper

**Result:** Code appears safe for execution
**Reason:** {reason}

---
*This PR has been automatically audited. A human reviewer may still want to verify the trading logic.*
"""
            if dry_run:
                logger.info(f"🏃 [DRY-RUN] Would approve PR #{pr.number}")
                return True
            
            # Add approval comment
            pr.create_issue_comment(comment)
            
            if auto_merge:
                # Check if mergeable
                for _ in range(5):
                    if pr.mergeable is not None:
                        break
                    time.sleep(2)
                    pr = repo.get_pull(pr.number)
                
                if pr.mergeable and pr.mergeable_state != 'dirty':
                    pr.merge(commit_message=f"🤖 Auto-merge: Gatekeeper approved #{pr.number}")
                    logger.info(f"✅ Merged PR #{pr.number}")
                else:
                    logger.warning(f"⚠️ PR #{pr.number} not mergeable: {pr.mergeable_state}")
            else:
                logger.info(f"✅ Approved PR #{pr.number} (manual merge required)")
            
            return True
            
        else:  # REJECT
            comment = f"""❌ **Security Audit: REJECTED**

🤖 Automated safety check by Moon Dev's PR Gatekeeper

**Result:** Potential security concern detected
**Reason:** {reason}

---
*This PR requires manual review before merging. Please address the security concerns or provide justification.*
"""
            if dry_run:
                logger.info(f"🏃 [DRY-RUN] Would reject PR #{pr.number}")
                return False
            
            # Add rejection comment
            pr.create_issue_comment(comment)
            logger.info(f"❌ Rejected PR #{pr.number}")
            
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to process PR #{pr.number}: {e}")
        return False


def main(dry_run: bool = False, auto_merge: bool = False):
    """Main processing loop"""
    logger.info("🌙 Moon Dev's PR Gatekeeper Starting...")
    
    if dry_run:
        logger.info("🏃 Running in DRY-RUN mode - no changes will be made")
    if auto_merge:
        logger.info("⚡ AUTO-MERGE enabled - approved PRs will be merged automatically")
    
    # Initialize clients
    gh, repo = init_github()
    
    # Get Jules PRs
    prs = get_jules_prs(repo)
    
    if not prs:
        logger.info("📭 No Jules PRs to audit")
        return
    
    approved = 0
    rejected = 0
    
    for pr in prs:
        # Skip draft PRs
        if pr.draft:
            logger.info(f"⏭️ Skipping draft PR #{pr.number}")
            continue
        
        # Check if already audited (has gatekeeper comment)
        comments = list(pr.get_issue_comments())
        gk_comments = [c for c in comments if "PR Gatekeeper" in c.body]
        
        if gk_comments:
            # Check if last audit was REJECTED and there are new commits since
            last_gk = gk_comments[-1]
            was_rejected = "REJECTED" in last_gk.body
            
            if was_rejected:
                # Check for new commits after the rejection
                commits = list(pr.get_commits())
                last_commit_date = commits[-1].commit.committer.date
                
                if last_commit_date > last_gk.created_at:
                    logger.info(f"🔄 PR #{pr.number} has new commits after rejection - re-auditing")
                else:
                    logger.info(f"⏭️ PR #{pr.number} rejected, no new commits, skipping")
                    continue
            else:
                # Was approved, skip
                logger.info(f"⏭️ PR #{pr.number} already approved, skipping")
                continue
        
        # Audit the PR
        verdict, reason = audit_pr(pr)
        
        # Process based on verdict
        if process_pr(repo, pr, verdict, reason, dry_run, auto_merge):
            approved += 1
        else:
            rejected += 1
        
        # Rate limiting
        time.sleep(2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Done!")
    logger.info(f"   ✅ Approved: {approved}")
    logger.info(f"   ❌ Rejected: {rejected}")
    logger.info(f"   📋 Total: {approved + rejected}")


if __name__ == "__main__":
    try:
        dry_run = "--dry-run" in sys.argv
        auto_merge = "--auto" in sys.argv
        main(dry_run=dry_run, auto_merge=auto_merge)
    except KeyboardInterrupt:
        logger.info("\n👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
