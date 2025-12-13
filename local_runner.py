#!/usr/bin/env python3
"""
Local Runner - Moon Dev Trading Factory
Executes strategies and manages results
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/local_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
STRATEGIES_DIR = Path('strategies')
RESULTS_DIR = Path('results')
PLOTS_DIR = RESULTS_DIR / 'plots'
LEADERBOARD_FILE = RESULTS_DIR / 'leaderboard.csv'
TEMP_RESULT_FILE = RESULTS_DIR / 'temp_result.json'
POLL_INTERVAL = 60  # seconds
STRATEGY_TIMEOUT = 300  # 5 minutes max execution


class LocalRunner:
    def __init__(self):
        """Initialize the local runner"""
        self.setup_directories()
        self.processed_strategies = self.load_processed_strategies()
        self.initialize_leaderboard()
        
    def setup_directories(self):
        """Create necessary directories"""
        STRATEGIES_DIR.mkdir(exist_ok=True)
        RESULTS_DIR.mkdir(exist_ok=True)
        PLOTS_DIR.mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        logger.info("Directories initialized")
        
    def load_processed_strategies(self) -> set:
        """Load set of already processed strategies"""
        if LEADERBOARD_FILE.exists():
            try:
                df = pd.read_csv(LEADERBOARD_FILE)
                processed = set(df['strategy_name'].unique())
                logger.info(f"Loaded {len(processed)} previously processed strategies")
                return processed
            except Exception as e:
                logger.warning(f"Error loading processed strategies: {e}")
                return set()
        return set()
        
    def initialize_leaderboard(self):
        """Create leaderboard CSV if it doesn't exist"""
        if not LEADERBOARD_FILE.exists():
            logger.warning("Leaderboard file not found - creating new one")
            # Create backup directory for future use
            backup_dir = RESULTS_DIR / 'backups'
            backup_dir.mkdir(exist_ok=True)
            df = pd.DataFrame(columns=[
                'timestamp',
                'strategy_name',
                'return_pct',
                'sharpe_ratio',
                'max_drawdown_pct',
                'win_rate_pct',
                'total_trades',
                'status'
            ])
            df.to_csv(LEADERBOARD_FILE, index=False)
            logger.info("Initialized leaderboard.csv")
    
    def backup_leaderboard(self):
        """Create a timestamped backup of the leaderboard"""
        if LEADERBOARD_FILE.exists():
            backup_dir = RESULTS_DIR / 'backups'
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f'leaderboard_{timestamp}.csv'
            shutil.copy(LEADERBOARD_FILE, backup_file)
            logger.info(f"Created leaderboard backup: {backup_file.name}")
            # Keep only last 10 backups
            backups = sorted(backup_dir.glob('leaderboard_*.csv'), reverse=True)
            for old_backup in backups[10:]:
                old_backup.unlink()

            
    def git_pull(self) -> bool:
        """Execute git pull to sync latest code"""
        try:
            result = subprocess.run(
                ['git', 'pull'],
                capture_output=True,
                text=True,
                timeout=120  # Increased from 30s to 120s
            )
            
            if result.returncode == 0:
                if 'Already up to date' not in result.stdout:
                    logger.info(f"Git pull successful: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"Git pull failed with return code {result.returncode}")
                logger.warning(f"STDOUT: {result.stdout}")
                logger.warning(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Git pull timed out after 120 seconds")
            return False
        except Exception as e:
            logger.error(f"Error during git pull: {e}")
            return False
            
    def detect_new_strategies(self) -> List[Path]:
        """Detect new or modified strategy files"""
        new_strategies = []
        
        for strategy_file in STRATEGIES_DIR.glob('*.py'):
            strategy_name = strategy_file.stem
            
            # Skip __init__.py and other utility files
            if strategy_name.startswith('__'):
                continue
                
            # Check if already processed
            if strategy_name not in self.processed_strategies:
                new_strategies.append(strategy_file)
                logger.info(f"Detected new strategy: {strategy_name}")
                
        return new_strategies
        
    def execute_strategy(self, strategy_file: Path) -> Optional[Dict]:
        """
        Execute a strategy file as subprocess
        
        Returns:
            Result dict or None if failed
        """
        strategy_name = strategy_file.stem
        logger.info(f"Executing strategy: {strategy_name}")
        
        try:
            # Clean up temp result file
            if TEMP_RESULT_FILE.exists():
                TEMP_RESULT_FILE.unlink()
                
            # Execute strategy with timeout
            result = subprocess.run(
                [sys.executable, str(strategy_file)],
                capture_output=True,
                text=True,
                timeout=STRATEGY_TIMEOUT,
                cwd=Path.cwd()
            )
            
            # Check for errors
            if result.returncode != 0:
                logger.error(f"Strategy {strategy_name} failed with error:\n{result.stderr}")
                return {
                    'strategy_name': strategy_name,
                    'status': 'ERROR',
                    'error': result.stderr[:500]
                }
                
            # Check if temp result was created
            if not TEMP_RESULT_FILE.exists():
                logger.warning(f"Strategy {strategy_name} did not create result file")
                return {
                    'strategy_name': strategy_name,
                    'status': 'NO_OUTPUT'
                }
                
            # Read results
            with open(TEMP_RESULT_FILE, 'r') as f:
                results = json.load(f)
                
            results['status'] = 'SUCCESS'
            logger.info(f"Strategy {strategy_name} executed successfully")
            logger.info(f"  Return: {results.get('return', 'N/A')}%")
            logger.info(f"  Sharpe: {results.get('sharpe', 'N/A')}")
            
            return results
            
        except subprocess.TimeoutExpired:
            logger.error(f"Strategy {strategy_name} timed out after {STRATEGY_TIMEOUT}s")
            return {
                'strategy_name': strategy_name,
                'status': 'TIMEOUT'
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse results for {strategy_name}: {e}")
            return {
                'strategy_name': strategy_name,
                'status': 'INVALID_JSON'
            }
            
        except Exception as e:
            logger.error(f"Error executing {strategy_name}: {e}")
            return {
                'strategy_name': strategy_name,
                'status': 'EXCEPTION',
                'error': str(e)
            }
            
    def harvest_results(self, results: Dict):
        """Add results to leaderboard"""
        try:
            # Prepare row
            row = {
                'timestamp': datetime.now().isoformat(),
                'strategy_name': results.get('strategy_name', 'unknown'),
                'return_pct': results.get('return', None),
                'sharpe_ratio': results.get('sharpe', None),
                'max_drawdown_pct': results.get('max_drawdown', None),
                'win_rate_pct': results.get('win_rate', None),
                'total_trades': results.get('total_trades', None),
                'status': results.get('status', 'UNKNOWN')
            }
            
            # Append to leaderboard
            df = pd.DataFrame([row])
            df.to_csv(LEADERBOARD_FILE, mode='a', header=False, index=False)
            
            logger.info(f"Added {results['strategy_name']} to leaderboard")
            
        except Exception as e:
            logger.error(f"Error harvesting results: {e}")
            
    def move_plot(self, strategy_name: str):
        """Move generated plot to plots directory"""
        try:
            # Backtesting.py typically creates plot as .html
            # Look for any recently modified HTML files
            html_files = list(Path.cwd().glob('*.html'))
            
            if not html_files:
                logger.warning(f"No plot found for {strategy_name}")
                return
                
            # Get most recently modified HTML
            latest_plot = max(html_files, key=lambda p: p.stat().st_mtime)
            
            # Move to plots directory
            dest = PLOTS_DIR / f"{strategy_name}.html"
            shutil.move(str(latest_plot), str(dest))
            
            logger.info(f"Moved plot to {dest}")
            
        except Exception as e:
            logger.error(f"Error moving plot for {strategy_name}: {e}")
            
    def run_loop(self):
        """Main execution loop"""
        logger.info("Local Runner started")
        
        while True:
            try:
                # Step 1: Git pull
                self.git_pull()
                
                # Step 2: Detect new strategies
                new_strategies = self.detect_new_strategies()
                
                # Step 3: Execute each new strategy
                for strategy_file in new_strategies:
                    strategy_name = strategy_file.stem
                    
                    # Execute
                    results = self.execute_strategy(strategy_file)
                    
                    if results:
                        # Harvest results
                        self.harvest_results(results)
                        
                        # Move plot if successful
                        if results.get('status') == 'SUCCESS':
                            self.move_plot(strategy_name)
                            
                        # Mark as processed
                        self.processed_strategies.add(strategy_name)
                        
                if new_strategies:
                    logger.info(f"Processed {len(new_strategies)} new strategies")
                    # Backup leaderboard after processing
                    self.backup_leaderboard()
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                
            # Wait before next cycle
            time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    try:
        runner = LocalRunner()
        runner.run_loop()
    except KeyboardInterrupt:
        logger.info("Local Runner stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
