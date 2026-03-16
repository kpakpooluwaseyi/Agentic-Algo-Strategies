"""
Quant Loop Orchestrator
=======================
Main entry point for the Autonomous Quant Factory.
Orchestrates the Researcher -> Developer -> Auditor recursive loop.
"""

import sys
import time
import argparse
import logging
from pathlib import Path

# Add root to python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.researcher_agent import ResearcherAgent
from src.agents.developer_agent import DeveloperAgent
from src.agents.auditor_agent import AuditorAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [ORCHESTRATOR] - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Orchestrator")

def main():
    parser = argparse.ArgumentParser(description="Run the Autonomous Quant Loop")
    parser.add_argument("--iterations", type=int, default=5, help="Maximum number of loop iterations")
    parser.add_argument("--cleanup", action="store_true", help="Clear previous context before starting")
    args = parser.parse_args()
    
    # Initialize Agents
    researcher = ResearcherAgent(verbose=True)
    developer = DeveloperAgent(verbose=True)
    auditor = AuditorAgent(verbose=True)
    
    # Validation file context
    feedback_file = Path("audit_feedback.md")
    thesis_file = Path("research_thesis.md")
    
    if args.cleanup:
        if feedback_file.exists(): feedback_file.unlink()
        if thesis_file.exists(): thesis_file.unlink()
        logger.info("🧹 Cleaned up previous context.")
        
    logger.info(f"🚀 Starting Autonomous Loop (Max Iterations: {args.iterations})")
    
    for i in range(1, args.iterations + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 LOOP ITERATION {i}/{args.iterations}")
        logger.info(f"{'='*60}\n")
        
        # 1. Researcher Step
        logger.info("🧠 STEP 1: RESEARCHER AGENT")
        if not researcher.run():
            logger.error("❌ Researcher failed to generate thesis. Aborting loop.")
            break
            
        # 2. Developer Step
        logger.info("👨‍💻 STEP 2: DEVELOPER AGENT")
        if not developer.run():
            logger.error("❌ Developer failed to implement strategy. Retrying Researcher...")
            # Create a fake feedback to force researcher to try again?
            # Or just continue loop which will trigger researcher again?
            # If developer failed, maybe thesis was bad.
            # Let's write a generic feedback:
            with open(feedback_file, "w") as f:
                f.write("# Audit Feedback: FAILED\nStatus: FAILED\n\nDeveloper Agent could not implement the thesis. Please simplify.")
            continue
            
        # 3. Auditor Step
        logger.info("🕵️ STEP 3: AUDITOR AGENT")
        # Auditor finds the latest strategy file automatically
        if auditor.run():
            logger.info("\n🎉 SUCCESSS!!! STrATEGY PASSED AUDIT!")
            logger.info("Check strategies/ folder for the validated file.")
            break
        else:
            logger.info("📉 Strategy failed audit. Pivoting for next iteration...")
            # Loop continues, Researcher will pick up audit_feedback.md
            
        time.sleep(2) # Brief pause beacuse robots need rest too (and file I/O safety)

    logger.info("\n🏁 Autonomous Loop Finished.")

if __name__ == "__main__":
    main()
