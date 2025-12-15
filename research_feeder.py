#!/usr/bin/env python3
"""
🌙 Moon Dev's Research Feeder
Extracts trading strategies from research inputs and creates GitHub Issues for Jules.

Uses Gemini 2.5 Flash for strategy extraction (~$0.10/1M tokens)

Supported input formats:
- PDF files (.pdf)
- Text files (.txt, .md)
- YouTube URLs (in a .txt file, one per line)

Usage:
    python research_feeder.py
    python research_feeder.py --dry-run  # Preview without creating issues
"""

import os
import sys
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import hashlib

from dotenv import load_dotenv
import google.generativeai as genai
from github import Github
import PyPDF2
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# Configuration
RESEARCH_INPUTS_DIR = Path("research_inputs")
PROCESSED_DIR = RESEARCH_INPUTS_DIR / ".processed"
LOGS_DIR = Path("logs")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "kpakpooluwaseyi/Agentic-Algo-Strategies")

# Gemini model - using 2.5 Flash for 1M context window
GEMINI_MODEL = "models/gemini-2.5-flash"

# Setup logging
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "research_feeder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Strategy extraction prompt
EXTRACTION_PROMPT = """You are an expert quantitative trading researcher. Analyze the following content and extract trading strategy specifications.

For each distinct trading strategy you find, output a structured specification with:

1. **Strategy Name**: A clear, descriptive name (snake_case for code compatibility)
2. **Strategy Type**: (e.g., momentum, mean-reversion, breakout, scalping, swing)
3. **Timeframe**: Recommended chart timeframe (1m, 5m, 15m, 1h, 4h, 1D)
4. **Instruments**: Applicable instruments (e.g., BTC, ETH, crypto, forex, stocks)

5. **Entry Rules**:
   - List all conditions that must be met to enter a trade
   - Be specific about indicator values, price levels, patterns

6. **Exit Rules**:
   - Take profit conditions
   - Stop loss conditions
   - Time-based exits if any

7. **Indicators Required**:
   - List all technical indicators needed (EMA, RSI, MACD, etc.)
   - Include specific periods/parameters

8. **Risk Management**:
   - Position sizing rules
   - Maximum risk per trade
   - Risk-reward ratio

9. **Additional Notes**:
   - Any special conditions or filters
   - Market regime considerations
   - Session timing if relevant

---
CONTENT TO ANALYZE:

{content}

---
Output each strategy as a separate section. If multiple strategies are found, separate them with "---".
If no clear trading strategy is found, output "NO_STRATEGY_FOUND" with an explanation.
"""

ISSUE_TEMPLATE = """## 🤖 Auto-Generated Strategy Request

**Source:** {source}
**Extracted:** {timestamp}

---

{strategy_content}

---

## Implementation Instructions

Please implement this strategy using the `backtesting.py` framework with the following requirements:

1. **File Location:** `strategies/{filename}.py`
2. **Data Path:** Use `data/BTC-USD-15m.csv` for backtesting
3. **Indicators:** Use `talib` or `pandas_ta` (NOT backtesting.py's built-in indicators)
4. **Template:** Follow existing strategies in the `strategies/` folder

### Required Components:
- Strategy class inheriting from `Strategy`
- Proper indicator initialization using `self.I()`
- Entry/exit logic in `next()` method
- Default parameters as class attributes
- Risk management (stop loss, take profit)

### Data Format:
```
datetime, open, high, low, close, volume
2023-01-01 00:00:00, 16531.83, 16532.69, 16509.11, 16510.82, 231.05
```

### Output:
- Run backtest with default parameters
- Print stats and create HTML plot
- Save results to `results/temp_result.json`
"""


def init_gemini():
    """Initialize Gemini client"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment!")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    logger.info(f"✅ Initialized Gemini model: {GEMINI_MODEL}")
    return model


def init_github():
    """Initialize GitHub client"""
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN not found in environment!")
    
    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(GITHUB_REPO)
    logger.info(f"✅ Connected to GitHub repo: {GITHUB_REPO}")
    return repo


def extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(video_id: str) -> Optional[str]:
    """Fetch transcript from YouTube video using new API"""
    try:
        # New YouTubeTranscriptApi v1.x uses instance-based API
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=['en'])
        # transcript is now a FetchedTranscript object, iterate to get text
        text = ' '.join([snippet.text for snippet in transcript])
        logger.info(f"📺 Fetched YouTube transcript: {len(text)} chars")
        return text
    except Exception as e:
        logger.error(f"❌ Failed to get YouTube transcript: {e}")
        return None


def extract_pdf_text(filepath: Path) -> Optional[str]:
    """Extract text from PDF file"""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        logger.info(f"📄 Extracted PDF text: {len(text)} chars from {filepath.name}")
        return text
    except Exception as e:
        logger.error(f"❌ Failed to extract PDF: {e}")
        return None


def read_text_file(filepath: Path) -> Optional[str]:
    """Read text from file, handling YouTube URLs"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if it's a list of YouTube URLs
        lines = content.strip().split('\n')
        youtube_content = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            video_id = extract_youtube_id(line)
            if video_id:
                logger.info(f"🎬 Found YouTube URL: {line}")
                transcript = get_youtube_transcript(video_id)
                if transcript:
                    youtube_content.append(f"[Video: {line}]\n{transcript}")
        
        if youtube_content:
            return "\n\n---\n\n".join(youtube_content)
        
        logger.info(f"📝 Read text file: {len(content)} chars from {filepath.name}")
        return content
        
    except Exception as e:
        logger.error(f"❌ Failed to read text file: {e}")
        return None


def get_content_hash(content: str) -> str:
    """Generate hash of content for deduplication"""
    return hashlib.md5(content.encode()).hexdigest()[:12]


def process_file(filepath: Path) -> Optional[str]:
    """Process a single file and return its content"""
    suffix = filepath.suffix.lower()
    
    if suffix == '.pdf':
        return extract_pdf_text(filepath)
    elif suffix in ['.txt', '.md']:
        return read_text_file(filepath)
    else:
        logger.warning(f"⚠️ Unsupported file type: {suffix}")
        return None


def extract_strategies(model, content: str) -> List[str]:
    """Use Gemini to extract trading strategies from content"""
    try:
        prompt = EXTRACTION_PROMPT.format(content=content[:100000])  # Limit content size
        
        logger.info(f"🤖 Sending {len(content)} chars to Gemini for analysis...")
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            logger.error("❌ Empty response from Gemini")
            return []
        
        result = response.text.strip()
        
        if "NO_STRATEGY_FOUND" in result:
            logger.warning(f"⚠️ No strategies found in content")
            return []
        
        # Split by separator if multiple strategies
        strategies = result.split("\n---\n")
        strategies = [s.strip() for s in strategies if s.strip()]
        
        logger.info(f"✅ Extracted {len(strategies)} strategy specification(s)")
        return strategies
        
    except Exception as e:
        logger.error(f"❌ Gemini extraction failed: {e}")
        return []


def create_github_issue(repo, strategy: str, source: str, dry_run: bool = False) -> bool:
    """Create a GitHub Issue for Jules to pick up"""
    try:
        # Generate filename from strategy content
        name_match = re.search(r'\*\*Strategy Name\*\*:\s*(.+)', strategy)
        if name_match:
            strategy_name = name_match.group(1).strip()
            filename = re.sub(r'[^a-z0-9_]', '_', strategy_name.lower())
        else:
            filename = f"strategy_{get_content_hash(strategy)}"
        
        title = f"🤖 [AUTO] Implement Strategy: {filename}"
        
        body = ISSUE_TEMPLATE.format(
            source=source,
            timestamp=datetime.now().isoformat(),
            strategy_content=strategy,
            filename=filename
        )
        
        if dry_run:
            logger.info(f"🏃 [DRY-RUN] Would create issue: {title}")
            logger.info(f"📝 Body preview:\n{body[:500]}...")
            return True
        
        # Check for existing issues with same title
        existing = list(repo.get_issues(state='open'))
        for issue in existing:
            if issue.title == title:
                logger.warning(f"⚠️ Issue already exists: #{issue.number}")
                return False
        
        issue = repo.create_issue(
            title=title,
            body=body,
            labels=["auto-generated", "strategy-request"]
        )
        
        logger.info(f"✅ Created GitHub Issue #{issue.number}: {title}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create GitHub issue: {e}")
        return False


def main(dry_run: bool = False):
    """Main processing loop"""
    logger.info("🌙 Moon Dev's Research Feeder Starting...")
    logger.info(f"📂 Looking for inputs in: {RESEARCH_INPUTS_DIR}")
    
    if dry_run:
        logger.info("🏃 Running in DRY-RUN mode - no issues will be created")
    
    # Initialize clients
    model = init_gemini()
    repo = init_github()
    
    # Create directories
    RESEARCH_INPUTS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    
    # Find input files
    input_files = []
    for pattern in ['*.pdf', '*.txt', '*.md']:
        input_files.extend(RESEARCH_INPUTS_DIR.glob(pattern))
    
    # Filter out processed marker files
    input_files = [f for f in input_files if not f.name.startswith('.')]
    
    if not input_files:
        logger.info("📭 No input files found in research_inputs/")
        return
    
    logger.info(f"📚 Found {len(input_files)} input file(s)")
    
    strategies_created = 0
    
    for filepath in input_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"📖 Processing: {filepath.name}")
        
        # Check if already processed
        processed_marker = PROCESSED_DIR / f"{filepath.name}.done"
        if processed_marker.exists():
            logger.info(f"⏭️ Already processed, skipping")
            continue
        
        # Extract content
        content = process_file(filepath)
        if not content:
            continue
        
        # Extract strategies using Gemini
        strategies = extract_strategies(model, content)
        
        # Create GitHub issues for each strategy
        for i, strategy in enumerate(strategies):
            source = f"{filepath.name}" + (f" (strategy {i+1})" if len(strategies) > 1 else "")
            if create_github_issue(repo, strategy, source, dry_run):
                strategies_created += 1
        
        # Mark as processed
        if not dry_run and strategies:
            processed_marker.touch()
            logger.info(f"✅ Marked {filepath.name} as processed")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Done! Created {strategies_created} GitHub issue(s)")
    
    if strategies_created > 0:
        logger.info(f"💡 Jules will pick up these issues and create PRs automatically")
        logger.info(f"🔗 View issues: https://github.com/{GITHUB_REPO}/issues")


if __name__ == "__main__":
    try:
        dry_run = "--dry-run" in sys.argv
        main(dry_run=dry_run)
    except KeyboardInterrupt:
        logger.info("\n👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
