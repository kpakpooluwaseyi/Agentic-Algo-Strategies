# Walkthrough 08: Swarm Pipeline Scripts Recreation

## Problem Summary
The `research_feeder.py` and `pr_gatekeeper.py` scripts were deleted before being committed to git. These scripts are essential for the "Cloud-Native" trading factory pipeline.

## Root Cause
Scripts were created in a previous session but not committed. Only log files remained as evidence they existed.

## Changes Made

### 1. Created `research_feeder.py`
**Purpose:** Extract trading strategies from research materials and create GitHub Issues for Jules.

**Key Features:**
- Reads PDFs, text files, and YouTube transcripts from `research_inputs/`
- Uses Gemini 2.5 Flash (1M token context) for strategy extraction
- Creates structured GitHub Issues with implementation instructions
- Deduplication to prevent duplicate issues
- Dry-run mode for testing

**Model:** `models/gemini-2.5-flash`

---

### 2. Created `pr_gatekeeper.py`
**Purpose:** Audit Jules PRs for security before merging.

**Key Features:**
- Monitors PRs from `feat/` and `feature/` branches
- Uses OpenRouter free tier (Qwen 3, Gemma 3, Llama 3.3)
- Checks for malware, infinite loops, suspicious imports
- Auto-approve safe PRs, reject unsafe ones with comments
- Dry-run and auto-merge modes

**Model:** OpenRouter free tier (rotating models)

---

## Live Test Results (2025-12-16)

### 🎉 Created 40 GitHub Issues (#269 - #308)

| Source PDF | Strategies Extracted |
|------------|---------------------|
| Fibonacci Trading by Carolyn Borden | 3 |
| Introduction to Quantitative Finance | 3 |
| Financial Derivatives Overview | 7 |
| Evidence-Based Technical Analysis | 5 |
| A Complete Guide to Volume Price Analysis | 2 |
| Peter Wyckoff - Psychology of Stock Market Timing | 3 |
| Quantitative Momentum Guide | 1 |
| Trendline Trading Strategy | 4 |
| Testing and Tuning Market Trading Systems | 2 |
| E-book 2 (Market Maker) | 6 |
| Beat The Market Maker by Steve Mauro | 2 |

**Example strategies:**
- `fibonacci_price_cluster_setup`
- `put_call_parity_arbitrage`
- `european_box_spread_arbitrage`
- `vpa_smart_money_reversal`
- `_mm_2nd_leg_m_w_reversal_`

---

## Architecture Flow

```
research_inputs/ → research_feeder.py → GitHub Issues → Jules Agent → PRs → pr_gatekeeper.py → Merge
                    (Gemini 2.5 Flash)                                        (OpenRouter)
```

---

## Usage

```bash
# Extract strategies and create issues
python research_feeder.py

# Dry-run mode (no issues created)
python research_feeder.py --dry-run

# Audit PRs for safety
python pr_gatekeeper.py

# Auto-merge safe PRs
python pr_gatekeeper.py --auto
```

---

## Files Created
- [research_feeder.py](file:///Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/research_feeder.py)
- [pr_gatekeeper.py](file:///Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/pr_gatekeeper.py)

## Environment Variables Required
- `GEMINI_API_KEY` - For research_feeder.py
- `OPENROUTER_API_KEY` - For pr_gatekeeper.py
- `GITHUB_TOKEN` - For both scripts
- `GITHUB_REPO` - Target repository
