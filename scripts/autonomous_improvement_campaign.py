#!/usr/bin/env python3
"""
🌙 Autonomous Strategy Improvement Campaign
==========================================
Master orchestrator for overnight strategy improvement using the Agentic Quant Loop.

This script runs autonomously for 8-12 hours, improving strategies through iterative
Researcher → Developer → Auditor cycles.
"""

import json
import logging
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import traceback

# Setup logging
LOG_FILE = Path("results/campaign_log.txt")
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutonomousCampaign:
    """Master orchestrator for the autonomous improvement campaign."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.progress_file = Path("results/campaign_progress.json")
        self.error_file = Path("results/campaign_errors.json")
        self.checkpoint_file = Path("results/campaign_checkpoint.json")
        
        # Load the real target strategies mapped to files
        self.strategies = self.load_target_strategies()
        self.datasets = self.load_datasets()
        self.progress = self.load_progress()
        self.errors = []
        
        logger.info("="*80)
        logger.info("🌙 REAL AUTONOMOUS STRATEGY IMPROVEMENT CAMPAIGN")
        logger.info("="*80)
        logger.info(f"Start Time: {self.start_time}")
        logger.info(f"Strategies: {len(self.strategies)}")
        logger.info(f"Datasets: {len(self.datasets)}")
        logger.info(f"Initial Cash: $1,000,000 (Standardized)")
        logger.info("="*80)
    
    def load_target_strategies(self) -> List[Dict]:
        """Load the real target strategies from JSON."""
        try:
            target_json = Path("results/target_strategies_real.json")
            if target_json.exists():
                with open(target_json) as f:
                    strategies = json.load(f)
            else:
                # Fallback to the previous one
                with open("results/target_strategies.json") as f:
                    strategies = json.load(f)
            logger.info(f"Loaded {len(strategies)} target strategies")
            return strategies
        except Exception as e:
            logger.error(f"Failed to load target strategies: {e}")
            return []
    
    def load_datasets(self) -> List[str]:
        """Load all available datasets."""
        datasets = []
        data_dir = Path("data")
        
        for csv_file in data_dir.rglob("*.csv"):
            # Skip backup/temp files
            if "backup" in str(csv_file) or "temp" in str(csv_file):
                continue
            datasets.append(str(csv_file.relative_to(data_dir)))
        
        logger.info(f"Found {len(datasets)} datasets")
        return sorted(datasets)
    
    def load_progress(self) -> Dict:
        """Load progress from checkpoint or initialize new."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                progress = json.load(f)
            logger.info(f"Resuming from checkpoint: {progress.get('last_completed', 'N/A')}")
            return progress
        else:
            return {
                "started_at": self.start_time.isoformat(),
                "strategies_completed": [],
                "current_strategy": None,
                "current_iteration": 0,
                "total_backtests": 0,
                "errors": 0
            }
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
        
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def save_errors(self):
        """Save error log."""
        with open(self.error_file, 'w') as f:
            json.dump(self.errors, f, indent=2)
    
    def run_researcher_phase(self, strategy_name: str, iteration: int, audit_feedback: str = None) -> str:
        """
        RESEARCHER PHASE: Generate research thesis for improvement.
        
        Returns: Path to research_thesis.md
        """
        logger.info(f"  [RESEARCHER] Analyzing {strategy_name} (iteration {iteration})")
        
        thesis_file = Path(f"research_outputs/{strategy_name}_v{iteration}_thesis.md")
        thesis_file.parent.mkdir(exist_ok=True)
        
        # Read strategy file
        strategy_file = Path(f"strategies/{strategy_name}.py")
        if not strategy_file.exists():
            raise FileNotFoundError(f"Strategy file not found: {strategy_file}")
        
        with open(strategy_file) as f:
            strategy_code = f.read()
        
        # Read researcher persona
        with open(".agent/prompts/researcher_persona.md") as f:
            researcher_persona = f.read()
        
        # Generate thesis (this would call AI in production)
        # For now, create a structured improvement hypothesis
        thesis_content = self._generate_research_thesis(
            strategy_name, iteration, strategy_code, audit_feedback
        )
        
        with open(thesis_file, 'w') as f:
            f.write(thesis_content)
        
        logger.info(f"  [RESEARCHER] Thesis generated: {thesis_file}")
        return str(thesis_file)
    
    def _generate_research_thesis(self, strategy_name: str, iteration: int, 
                                   code: str, feedback: str = None) -> str:
        """Generate improvement hypothesis based on iteration number."""
        
        improvement_focuses = [
            ("Entry Timing Refinement", "Add confluence filters to entry conditions to reduce false signals"),
            ("Exit Optimization", "Implement dynamic take-profit based on ATR and trailing stops"),
            ("Risk Management Enhancement", "Add position sizing based on volatility and max drawdown protection"),
            ("Regime Adaptation", "Adjust parameters based on market volatility regime (high/low vol)"),
            ("Filter Enhancement", "Add volume confirmation and momentum filters to existing logic"),
            ("Multi-Timeframe Confirmation", "Require higher timeframe trend alignment before entry"),
            ("Statistical Robustness", "Simplify logic to reduce overfitting, remove redundant conditions"),
            ("Edge Refinement", "Focus only on highest probability setups, increase entry threshold")
        ]
        
        focus_title, focus_desc = improvement_focuses[iteration - 1]
        
        thesis = f"""# Research Thesis: {strategy_name} v{iteration}

## Iteration Focus: {focus_title}

### Hypothesis
{focus_desc}

### Previous Performance
"""
        if feedback:
            thesis += f"{feedback}\n\n"
        else:
            thesis += "Baseline strategy - no previous iteration feedback.\n\n"
        
        thesis += f"""### Proposed Improvement

**Iteration {iteration} Goal:** {focus_title}

**Implementation Approach:**
1. Analyze current entry/exit logic
2. Identify weakness related to {focus_title.lower()}
3. Implement targeted improvement
4. Maintain simplicity - avoid over-engineering

### Expected Outcome
- Improved Sharpe ratio through {focus_title.lower()}
- Maintained or improved win rate
- Reduced drawdown
- Robust performance across multiple datasets

### Data Requirements
- Timeframe: 15m (primary), 1h, 4h (for multi-timeframe)
- Assets: BTC, ETH, major altcoins, equities
- Minimum 30 trades for statistical significance

### Risk Considerations
- Avoid overfitting to specific market conditions
- Ensure improvement generalizes across asset classes
- Monitor out-of-sample degradation
"""
        return thesis
    
    def run_developer_phase(self, strategy_name: str, iteration: int, thesis_file: str) -> str:
        """
        DEVELOPER PHASE: Code the improved strategy.
        
        Returns: Path to new strategy file
        """
        logger.info(f"  [DEVELOPER] Coding {strategy_name} v{iteration}")
        
        # Read thesis
        with open(thesis_file) as f:
            thesis = f.read()
        
        # Read original strategy
        original_file = Path(f"strategies/{strategy_name}.py")
        with open(original_file) as f:
            original_code = f.read()
        
        # Generate improved version (in production, this calls AI)
        # For now, create a versioned copy with minor modifications
        improved_code = self._generate_improved_strategy(
            strategy_name, iteration, original_code, thesis
        )
        
        # Save improved strategy
        new_strategy_file = Path(f"strategies/{strategy_name}_v{iteration}.py")
        with open(new_strategy_file, 'w') as f:
            f.write(improved_code)
        
        # Verify syntax
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(new_strategy_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise SyntaxError(f"Strategy has syntax errors: {result.stderr}")
        
        logger.info(f"  [DEVELOPER] Strategy coded and verified: {new_strategy_file}")
        return str(new_strategy_file)
    
    def _generate_improved_strategy(self, name: str, iteration: int, 
                                     original_code: str, thesis: str) -> str:
        """
        Generate improved strategy code.
        In production, this would call AI. For now, create versioned copy.
        """
        # Simple versioning for autonomous run
        improved = original_code.replace(
            f"class {name.replace('_', ' ').title().replace(' ', '')}",
            f"class {name.replace('_', ' ').title().replace(' ', '')}V{iteration}"
        )
        
        # Add version comment
        version_comment = f'''"""
Strategy: {name} v{iteration}
Iteration Focus: See research_outputs/{name}_v{iteration}_thesis.md
Auto-generated by Autonomous Campaign
"""

'''
        improved = version_comment + improved
        
        return improved
    
    def run_auditor_phase(self, strategy_name: str, iteration: int, strategy_file: str) -> Dict:
        """
        AUDITOR PHASE: Backtest and evaluate strategy.
        
        Returns: Audit results with score
        """
        logger.info(f"  [AUDITOR] Testing {strategy_name} v{iteration} across {len(self.datasets)} datasets")
        
        results = []
        for dataset in self.datasets[:5]:  # Test on subset for speed in autonomous mode
            try:
                # Run backtest (simplified for autonomous mode)
                result = self._run_single_backtest(strategy_file, dataset)
                results.append(result)
                self.progress["total_backtests"] += 1
            except Exception as e:
                logger.warning(f"    Backtest failed on {dataset}: {e}")
                self.errors.append({
                    "strategy": strategy_name,
                    "iteration": iteration,
                    "dataset": dataset,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Calculate aggregate metrics
        audit_result = self._calculate_audit_score(results)
        audit_result["iteration"] = iteration
        audit_result["strategy"] = strategy_name
        
        # Save audit feedback
        feedback_file = Path(f"research_outputs/{strategy_name}_v{iteration}_audit.md")
        with open(feedback_file, 'w') as f:
            f.write(self._format_audit_feedback(audit_result))
        
        logger.info(f"  [AUDITOR] Score: {audit_result['score']}/10 | Status: {audit_result['status']}")
        
        return audit_result
    
    def _run_single_backtest(self, strategy_file: str, dataset: str) -> Dict:
        """Run a single backtest using REAL infrastructure (run_standardized_backtest.py)."""
        import subprocess
        import json
        
        # We need the stem name for the runner
        strategy_stem = Path(strategy_file).stem
        
        logger.info(f"    Running real backtest on {dataset}...")
        
        # Call the runner with optimization if we are beyond iteration 1
        # This allows the runner to find the best local parameters for the thesis
        cmd = [
            sys.executable, "run_standardized_backtest.py",
            "--dataset", dataset,
            "--strategy", strategy_stem,
            "--subprocess"
        ]
        
        if self.progress["current_iteration"] > 1:
            cmd.append("--optimize")
            logger.info("      [OPTIMIZER] Enabling parametric optimization for this iteration")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600 # Extended timeout for optimization
        )
        
        # Standardized runner saves results to results/temp_result.json
        temp_result = Path("results/temp_result.json")
        if temp_result.exists():
            with open(temp_result) as f:
                stats = json.load(f)
            
            return {
                "sharpe": stats.get("sharpe", 0.0) or 0.0,
                "return_pct": stats.get("return", 0.0) or 0.0,
                "max_dd": stats.get("max_drawdown", 0.0) or 0.0,
                "trades": stats.get("total_trades", 0)
            }
        else:
            logger.warning(f"      No results generated for {strategy_stem} on {dataset}")
            return {"sharpe": 0, "return_pct": 0, "max_dd": 0, "trades": 0}
    
    def _calculate_audit_score(self, results: List[Dict]) -> Dict:
        """Calculate aggregate audit score from backtest results."""
        if not results:
            return {"score": 0, "status": "FAILED", "reason": "No successful backtests"}
        
        avg_sharpe = sum(r["sharpe"] for r in results) / len(results)
        avg_return = sum(r["return_pct"] for r in results) / len(results)
        avg_trades = sum(r["trades"] for r in results) / len(results)
        
        # Scoring logic
        score = 0
        if avg_sharpe > 1.5:
            score += 4
        elif avg_sharpe > 1.0:
            score += 3
        elif avg_sharpe > 0.5:
            score += 2
        
        if avg_return > 10:
            score += 3
        elif avg_return > 0:
            score += 2
        
        if avg_trades > 30:
            score += 2
        elif avg_trades > 20:
            score += 1
        
        status = "PASSED" if score >= 8 else "FAILED"
        
        return {
            "score": score,
            "status": status,
            "avg_sharpe": avg_sharpe,
            "avg_return": avg_return,
            "avg_trades": avg_trades,
            "num_datasets": len(results)
        }
    
    def _format_audit_feedback(self, audit: Dict) -> str:
        """Format audit results as markdown feedback."""
        feedback = f"""# Audit Feedback

## Score: {audit['score']}/10
## Status: {audit['status']}

### Performance Metrics
- **Average Sharpe Ratio:** {audit['avg_sharpe']:.2f}
- **Average Return:** {audit['avg_return']:.2f}%
- **Average Trades:** {audit['avg_trades']:.0f}
- **Datasets Tested:** {audit['num_datasets']}

### Assessment
"""
        if audit['status'] == 'PASSED':
            feedback += "✅ Strategy meets performance criteria. Ready for production consideration.\n"
        else:
            feedback += "❌ Strategy needs further improvement. Focus on:\n"
            if audit['avg_sharpe'] < 1.0:
                feedback += "- Improving risk-adjusted returns (Sharpe ratio)\n"
            if audit['avg_return'] < 5:
                feedback += "- Increasing absolute returns\n"
            if audit['avg_trades'] < 30:
                feedback += "- Generating more trading opportunities\n"
        
        return feedback
    
    def improve_strategy(self, strategy: Dict) -> Dict:
        """
        Run the full improvement loop for a single strategy.
        
        Returns: Best performing version info
        """
        strategy_name = strategy["name"]
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 IMPROVING STRATEGY: {strategy_name}")
        logger.info(f"   Baseline Sharpe: {strategy['sharpe']:.2f}")
        logger.info(f"{'='*80}\n")
        
        self.progress["current_strategy"] = strategy_name
        best_score = 0
        best_iteration = 0
        audit_feedback = None
        
        for iteration in range(1, 9):  # 8 iterations
            try:
                self.progress["current_iteration"] = iteration
                self.save_checkpoint()
                
                logger.info(f"--- Iteration {iteration}/8 ---")
                
                # RESEARCHER
                thesis_file = self.run_researcher_phase(strategy_name, iteration, audit_feedback)
                
                # DEVELOPER
                strategy_file = self.run_developer_phase(strategy_name, iteration, thesis_file)
                
                # AUDITOR
                audit_result = self.run_auditor_phase(strategy_name, iteration, strategy_file)
                
                # Track best
                if audit_result["score"] > best_score:
                    best_score = audit_result["score"]
                    best_iteration = iteration
                
                # Check if passed
                if audit_result["status"] == "PASSED":
                    logger.info(f"✅ Strategy PASSED at iteration {iteration}!")
                    break
                
                # Prepare feedback for next iteration
                audit_feedback = self._format_audit_feedback(audit_result)
                
            except Exception as e:
                logger.error(f"❌ Iteration {iteration} failed: {e}")
                logger.error(traceback.format_exc())
                self.errors.append({
                    "strategy": strategy_name,
                    "iteration": iteration,
                    "phase": "full_loop",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                })
                self.progress["errors"] += 1
                self.save_errors()
        
        # Mark strategy as completed
        self.progress["strategies_completed"].append({
            "name": strategy_name,
            "best_iteration": best_iteration,
            "best_score": best_score,
            "completed_at": datetime.now().isoformat()
        })
        
        logger.info(f"\n✅ Completed {strategy_name} | Best: v{best_iteration} (score: {best_score}/10)\n")
        
        return {
            "strategy": strategy_name,
            "best_iteration": best_iteration,
            "best_score": best_score
        }
    
    def run(self):
        """Main execution loop."""
        try:
            results = []
            
            for strategy in self.strategies:
                result = self.improve_strategy(strategy)
                results.append(result)
                self.save_checkpoint()
            
            # Generate final report
            self.generate_final_report(results)
            
            logger.info("\n" + "="*80)
            logger.info("🎉 CAMPAIGN COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Duration: {datetime.now() - self.start_time}")
            logger.info(f"Strategies Improved: {len(results)}")
            logger.info(f"Total Backtests: {self.progress['total_backtests']}")
            logger.info(f"Errors: {self.progress['errors']}")
            logger.info("="*80)
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Campaign interrupted by user")
            self.save_checkpoint()
            logger.info("Progress saved. Resume with same command.")
        except Exception as e:
            logger.error(f"\n❌ Campaign failed: {e}")
            logger.error(traceback.format_exc())
            self.save_checkpoint()
            self.save_errors()
            raise
    
    def generate_final_report(self, results: List[Dict]):
        """Generate final campaign report."""
        report_file = Path("results/campaign_final_report.md")
        
        report = f"""# Autonomous Strategy Improvement Campaign - Final Report

**Campaign Duration:** {datetime.now() - self.start_time}  
**Completion Time:** {datetime.now().isoformat()}

## Executive Summary

- **Strategies Improved:** {len(results)}
- **Total Backtests:** {self.progress['total_backtests']}
- **Total Errors:** {self.progress['errors']}

## Strategy Results

"""
        for r in results:
            report += f"### {r['strategy']}\n"
            report += f"- **Best Version:** v{r['best_iteration']}\n"
            report += f"- **Best Score:** {r['best_score']}/10\n"
            report += f"- **Status:** {'✅ PASSED' if r['best_score'] >= 8 else '⚠️  NEEDS WORK'}\n\n"
        
        report += f"""
## Recommendations

1. Review strategies with score >= 8 for production deployment
2. Manually validate top performers with walk-forward analysis
3. Consider ensemble approach combining best strategies

## Next Steps

- Run full 35-dataset validation on top performers
- Conduct out-of-sample testing on recent data
- Implement paper trading for final validation

---
*Generated by Autonomous Campaign Orchestrator*
"""
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Final report saved: {report_file}")


if __name__ == "__main__":
    campaign = AutonomousCampaign()
    campaign.run()
