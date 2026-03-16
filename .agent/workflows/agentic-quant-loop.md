---
description: Run the Human-in-the-Loop Agentic Quant Factory
---

# Agentic Quant Loop

This workflow replaces the Python script orchestration with a direct Agentic execution model.
**YOU (the AI)** are the computer. You will impersonate the agents using your internal models.

## Phase 1: Researcher (Claude 4.5 Opus)
1.  Read `.agent/prompts/researcher_persona.md`.
2.  Read `audit_feedback.md` (if exists) and `research_inputs/`.
3.  **THINK** deeply about market structure.
4.  **WRITE** `research_thesis.md` with a novel trading hypothesis.

## Phase 2: Developer (Gemini 3.0 Pro)
1.  Read `.agent/prompts/developer_persona.md`.
2.  Read `research_thesis.md`.
3.  **CODE** the strategy in `strategies/{StrategyName}.py`.
4.  **VERIFY** syntax (run `python -m py_compile strategies/{StrategyName}.py`).

## Phase 3: Auditor (Claude 4.5 Sonnet)
1.  Read `.agent/prompts/auditor_persona.md`.
2.  **EXECUTE** the strategy: `python strategies/{StrategyName}.py`.
3.  **ANALYZE** the output (Sharpe, Drawdown, Trades).
4.  **WRITE** `audit_feedback.md` with Score, Pass/Fail status, and Critique.

## Phase 4: Loop
-   If **PASSED** (Score >= 8): Congratulate user and stop.
-   If **FAILED** (Score < 8):  Go to Phase 1 (Researcher) and use feedback to improve.
-   Repeat until success or max 3 iterations.
