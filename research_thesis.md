# Research Thesis: BTC Quantile Support/Resistance with Trend Filter

**Date:** 2026-02-10
**Researcher:** Claude 4.5 Opus (Agentic)
**Version:** 1.1

---

## Hypothesis

> **Quantile-based support/resistance levels, when combined with a trend filter, identify high-probability mean reversion zones in BTC.**

The 20th percentile of recent lows defines "support" and the 80th percentile of recent highs defines "resistance." By only taking long trades above EMA200 (bullish regime) and short trades below EMA200 (bearish regime), we filter out counter-trend trades that have historically failed.

**Edge Explanation:**
- Quantile levels are adaptive—they adjust to volatility automatically.
- Trend filter prevents fighting strong directional moves.
- 15m timeframe captures intraday mean reversion in crypto's continuous market.

---

## Data Needs

| Parameter | Value |
|-----------|-------|
| Asset | BTC-USDT |
| Timeframe | 15m |
| Lookback | 200 bars (~50 hours) |
| Dataset | `BTC-USDT_15m_160weeks.csv` |

---

## Entry Rules

### Long Entry
```
Price <= Support * (1 + Buffer)  AND  Price > EMA200
```
Where:
- `Support = np.percentile(Low[-200:], 20)`
- `Buffer = 0.003` (0.3%)

### Short Entry
```
Price >= Resistance * (1 - Buffer)  AND  Price < EMA200
```
Where:
- `Resistance = np.percentile(High[-200:], 80)`

---

## Exit Rules

| Type | Rule |
|------|------|
| Take Profit | Dynamic flip (opposite signal triggered) |
| Stop Loss | 1.5% from entry |

---

## Risk Management

| Parameter | Value |
|-----------|-------|
| Risk per trade | 1.5% of equity |
| Maximum position | 30% of equity |
| Leverage | None (spot) |

---

## Expected Performance

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 1.0 |
| Win Rate | > 50% |
| Trade Count | > 30 (3 years) |
| Max Drawdown | < 20% |

---

## Implementation Notes

1. Use `np.percentile()` for quantile calculation (equivalent to TradingView's `ta.percentile_linear_interpolation`).
2. EMA200 calculated on Close prices.
3. Entry buffer prevents false breakouts.
4. Dynamic flipping = close position when opposite signal fires.
