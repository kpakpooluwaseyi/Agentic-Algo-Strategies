#!/usr/bin/env python3
"""
🌙 Moon Dev's PR Gatekeeper
Audits Pull Requests from Jules for safety before merging.

Uses Gemini 3.0 Flash for high-fidelity code auditing.

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
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configuration
LOGS_DIR = Path("logs")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "kpakpooluwaseyi/Agentic-Algo-Strategies")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini configuration
GEMINI_MODEL = "models/gemini-3-flash-preview"

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

# Safety audit prompt
AUDIT_PROMPT = """You are a security-focused code reviewer auditing auto-generated trading strategy code.

Analyze the following code diff and check for:

1. **CRITICAL SECURITY ISSUES** (REJECT immediately):
   - Malware patterns (backdoors, data exfiltration)
   - Network requests to unknown servers
   - File system access outside project directory
   - Execution of arbitrary code (eval, exec, compile)
   - Subprocess calls for non-trading purposes
   - Credential theft attempts

2. **DANGEROUS PATTERNS** (REJECT):
   - Infinite loops without proper exit conditions
   - Memory exhaustion (unbounded lists/dicts)
   - Disk filling (unbounded file writes)
   - API abuse (rapid-fire requests without limits)

3. **SUSPICIOUS BUT ACCEPTABLE** (WARN but APPROVE):
   - Using os.path for file handling (normal)
   - HTTP requests to known data APIs (Binance, CoinGecko, etc.)
   - subprocess for running backtests

4. **EXPECTED CODE** (APPROVE):
   - Strategy class definitions
   - Indicator calculations (talib, pandas_ta)
   - Backtest setup and execution
   - Results logging and plotting

---
CODE DIFF TO AUDIT:

{diff_content}

---
Respond with EXACTLY one of these formats:

If SAFE:
```
VERDICT: APPROVE
REASON: [Brief explanation of why the code is safe]
```

If UNSAFE:
```
VERDICT: REJECT
REASON: [Specific security concern found]
CODE_LOCATION: [Line or section with the issue]
```

If unclear, err on the side of caution and REJECT.
"""


def init_github():
    """Initialize GitHub client"""
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN not found in environment!")
    
    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(GITHUB_REPO)
    logger.info(f"✅ Connected to GitHub repo: {GITHUB_REPO}")
    return gh, repo


def call_gemini(prompt: str) -> Optional[str]:
    """Call Gemini API with the given prompt"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment!")
    
    try:
        logger.info(f"🤖 Calling Gemini model: {GEMINI_MODEL}")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent judgments
                max_output_tokens=1000,
            )
        )
        
        if response and response.text:
            logger.info(f"✅ Got response from {GEMINI_MODEL}")
            return response.text
        else:
            logger.error(f"❌ Gemini returned empty response")
            return None
                
    except Exception as e:
        logger.error(f"❌ Gemini call failed: {e}")
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
    """Get the diff content of a PR"""
    try:
        # Get files changed in the PR
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
    """Parse the verdict from LLM response"""
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
    
    # Send to Gemini for detailed analysis
    prompt = AUDIT_PROMPT.format(diff_content=diff[:50000])  # Limit size
    response = call_gemini(prompt)
    
    if not response:
        return "REJECT", "Gemini API unavailable - manual review required"
    
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
        already_audited = any("PR Gatekeeper" in c.body for c in comments)
        
        if already_audited:
            logger.info(f"⏭️ PR #{pr.number} already audited, skipping")
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
