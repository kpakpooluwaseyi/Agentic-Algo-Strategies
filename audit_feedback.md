# Audit Report: BTC Quantile SR Strategy (Final)

**Date:** 2026-02-10
**Auditor:** Claude 4.5 Sonnet (Agentic)
**Status:** **FAILED AFTER 3 ITERATIONS**

---

## Iteration Summary

| Version | Return | Sharpe | Trades | Win Rate | Key Changes |
|---------|--------|--------|--------|----------|-------------|
| V1 | +2.92% | 0.094 | 213 | 9.4% | Original (no TP) |
| V2 | -1.50% | -0.982 | 3 | 0.0% | Tight buffer (over-filtered) |
| V3 | -18.30% | -1.158 | 234 | 20.1% | Keep entries + add TP |

---

## Root Cause Analysis

The strategy has a **fundamental conceptual flaw**: Quantile-based support/resistance on 15m BTC does NOT identify high-probability reversal zones.

### Why It Fails

1. **Quantile levels are backward-looking**: The 20th/80th percentile of the last 200 bars tells you where price WAS, not where it will reverse.

2. **No confirmation**: The strategy enters immediately when price touches the level. Without momentum confirmation (RSI, MACD), it catches falling knives.

3. **The "positive return paradox"**: V1 showed +2.92% return with 9.4% win rate. This means the few winners were MASSIVE (dynamic flipping to opposite extreme). But this is not a tradeable edge—it's random outlier dependency.

4. **Adding TP made it worse**: V3 with proper 2:1 R:R take profit immediately showed the strategy's true nature: random entries = random results.

---

## Verdict

**HYPOTHESIS INVALIDATED**: Quantile percentile-based S/R with EMA trend filter does NOT provide edge on 15m BTC.

---

## Recommendations for Next Cycle

If you want to continue research on this approach:

1. **Add momentum confirmation**: RSI < 30 OR MACD histogram positive before entry
2. **Use higher timeframe**: 1h or 4h may have cleaner S/R levels
3. **Replace quantile with key levels**: Prior swing high/low, round numbers, or volume profile

**Alternative direction**: Abandon mean reversion entirely. The "Weighted Signals V2" strategy (+2.62%, 60% WR) from the earlier session is proven profitable on 15m BTC.

---

## Files Created

| File | Purpose |
|------|---------|
| `strategies/BTCQuantileSR.py` | V1 original |
| `strategies/BTCQuantileSR_V2.py` | V2 (over-filtered) |
| `strategies/BTCQuantileSR_V3.py` | V3 final iteration |
| `research_thesis.md` | Research hypothesis |
| `audit_feedback.md` | This report |
