#!/usr/bin/env python3
"""
🌙 Moon Dev's Overseer
====================
The central orchestration layer for the agentic swarm.
Manages the lifecycle of Feeder, Gatekeeper, and Runner using asyncio.

Usage:
    python overseer.py          # Run all agents in parallel
    python overseer.py --feeder # Run only the research feeder
    python overseer.py --runner # Run only the local runner
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'overseer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Overseer")

# Agent intervals (in seconds)
FEEDER_INTERVAL = 600   # 10 minutes
GATEKEEPER_INTERVAL = 60  # 1 minute
RUNNER_INTERVAL = 60     # 1 minute


async def run_feeder(interval: int = FEEDER_INTERVAL):
    """Loop for research_feeder.py"""
    logger.info("🔬 Research Feeder loop initialized")
    while True:
        try:
            logger.info("🏃 Running Research Feeder...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, "research_feeder.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"Feeder failed: {stderr.decode()[:500]}")
            else:
                logger.info("✅ Feeder cycle complete")
            
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Feeder loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in Feeder loop: {e}")
            await asyncio.sleep(60)


async def run_gatekeeper(interval: int = GATEKEEPER_INTERVAL):
    """Loop for pr_gatekeeper.py"""
    logger.info("🛡️ PR Gatekeeper loop initialized")
    while True:
        try:
            logger.info("🏃 Running PR Gatekeeper...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, "pr_gatekeeper.py", "--auto",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"Gatekeeper failed: {stderr.decode()[:500]}")
            else:
                logger.info("✅ Gatekeeper cycle complete")
            
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Gatekeeper loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in Gatekeeper loop: {e}")
            await asyncio.sleep(60)


async def run_runner(interval: int = RUNNER_INTERVAL):
    """Loop for local_runner.py (single cycle mode)"""
    logger.info("🏭 Local Runner loop initialized")
    while True:
        try:
            logger.info("🏃 Running Local Runner cycle...")
            # Run local_runner in single-cycle mode (we handle the loop here)
            process = await asyncio.create_subprocess_exec(
                sys.executable, "local_runner.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"Runner failed: {stderr.decode()[:500]}")
            else:
                logger.info("✅ Runner cycle complete")
            
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Runner loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in Runner loop: {e}")
            await asyncio.sleep(60)


async def main():
    logger.info("=" * 60)
    logger.info("🌙 Moon Dev Overseer Activated")
    logger.info("=" * 60)
    logger.info(f"📂 Project Root: {PROJECT_ROOT}")
    logger.info(f"⏰ Feeder Interval: {FEEDER_INTERVAL}s")
    logger.info(f"⏰ Gatekeeper Interval: {GATEKEEPER_INTERVAL}s")
    logger.info(f"⏰ Runner Interval: {RUNNER_INTERVAL}s")
    logger.info("=" * 60)
    
    # Parse command line args for selective agent running
    args = sys.argv[1:]
    
    tasks = []
    if not args or '--all' in args:
        # Run all agents
        tasks = [
            run_feeder(),
            run_gatekeeper(),
            run_runner()
        ]
        logger.info("🚀 Starting ALL agents in parallel...")
    else:
        if '--feeder' in args:
            tasks.append(run_feeder())
            logger.info("🔬 Starting Research Feeder only...")
        if '--gatekeeper' in args:
            tasks.append(run_gatekeeper())
            logger.info("🛡️ Starting PR Gatekeeper only...")
        if '--runner' in args:
            tasks.append(run_runner())
            logger.info("🏭 Starting Local Runner only...")
    
    if not tasks:
        logger.error("No agents specified. Use --all, --feeder, --gatekeeper, or --runner")
        return
    
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("👋 Overseer shutting down gracefully...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Moon Dev Overseer stopped.")
