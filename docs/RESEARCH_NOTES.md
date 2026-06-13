[🇬🇧 English](RESEARCH_NOTES.md) | [🇫🇷 Français](RESEARCH_NOTES.fr.md)

# Experimental plan (DOE) — Analyseur crypto
**Pre-registered on 2026-05-30. The hypotheses and criteria below are frozen BEFORE looking at the holdout.**

## Guiding principle
We have refuted 4 apparent "edges" (ML gate 1.88; detector+filter 45d; 4h 150d PF 2.04; feature filters) — all = artefacts of **direction/regime/tail**. The risk now is no longer missing an edge, it is **over-testing ourselves** by re-analyzing the same data in a loop. This DOE enforces: pre-registered hypotheses, locked holdout touched ONCE, multiple-comparison control, bootstrap confidence intervals, regime stratification, tail robustness.

## Assets (all generated data)
- `dataset_21d.csv` (224 trades, BEAR, multi-TF) — regime filter crippled
- `dataset_45d.csv` (578, ~balanced, multi-TF) — filter active
- `dataset_90d_compromised.csv` (259) — filter OFF (5d timeline), used only as counter-example
- current `dataset.csv` = **4h / 150d / 947 trades / filter active** — the deepest and cleanest
- Tools: `build_ml_dataset.py` (faithful backfill + enrichment), `evaluate.py` (purged walk-forward), regime reconstruction (`build_regime_timeline`/`_regime_at`)
- Hard limit: only the **4h** is deeply backtestable (~250d); 1h/15m capped (~45-60d) — see deep-fetch bug.

## Non-negotiable statistical rules
1. **Metric = PF / expectancy AFTER COSTS (0.2% and 0.4% round-trip), stratified by regime (BULL/BEAR/RANGE), with 95% bootstrap CI.** Never a point estimate alone.
2. **Tail robustness**: every result is recomputed by removing crash days (jackknife). An edge that does not survive the jackknife is not an edge.
3. **Locked cross-sectional holdout**: symbols split into DISCOVERY (13) / HOLDOUT (7, randomly drawn, frozen). Hypotheses are formed/tuned on DISCOVERY only. HOLDOUT is touched ONCE, at the end.
4. **Purged walk-forward** (embargo) within DISCOVERY for temporal robustness.
5. **Multiple-testing correction**: if we test N factors, the bar goes up (Bonferroni/FDR). We COUNT and LOG every hypothesis tested.
6. **Mandatory benchmarks**: every edge is measured against (a) matched RANDOM entries, (b) take-all, (c) pure directional beta (follow the regime).

## Phase 0 — Consolidation & splits
- Merge all datasets into a tagged master (symbol, TF, reconstructed regime, side, run, crash days).
- Draw and FREEZE the DISCOVERY/HOLDOUT symbol split.
- Tag crash days (e.g. Feb 03-05) for the jackknife.

## Phase 1 — FOUNDATIONAL TEST (GATE): does the detector beat random?
**H1:** the detector entries have an OOS expectancy > MATCHED random entries (same symbol, same side, same stop%/target%, same regime; only the entry CANDLE differs).
- Answer by regime, after costs, bootstrap CI on the detector−random difference.
- **GATE:** if the difference CI overlaps 0 across all regimes → patterns add NOTHING → we stop the "patterns" branch and pivot (Phase 3bis). If > 0 in ≥1 regime stably → we continue.

## Phase 2 — Decompose BETA vs ALPHA (GATE)
**H2:** after removing directional beta (return from "following the regime") AND crash days, a positive expectancy remains.
- Beta baseline = regime-matched random entries (from Phase 1).
- Alpha = detector − beta, crash jackknife, bootstrap CI.
- **GATE:** residual alpha CI > 0? Otherwise, the system = directional beta (no alpha) → strategic decision (Phase 5).

## Phase 3 — FACTORIAL DOE of structural features (if Phase 2 passes)
ON/OFF factors (machinery already in place): `confirmed S/R-break (HTF close + volume)`, `aligned BOS/CHoCH`, `vol_ratio`, `relative strength vs BTC`, `HTF S/R distance (ATR)`.
- 2^k factorial plan on DISCOVERY → regression/ANOVA to isolate the factors (and interactions) that significantly move expectancy.
- Multiple-testing correction. Each surviving factor → validated ONCE on HOLDOUT.

## Phase 3bis (if Phase 1/2 fail) — Pivot
Test a paradigm with actual empirical support: **cross-sectional momentum / relative strength vs BTC** (rank symbols, long the strong / short the weak), independent of chart patterns.

## Phase 4 — DOE of EXITS/RISK
Factors: stop ATR multiple, target RR, partial TP, trailing, time-stop, regime/vol-conditional sizing. Factorial on DISCOVERY → HOLDOUT validation.

## Phase 5 — Locked FINAL VERDICT
Best config from Phases 3-4 → ONE evaluation on HOLDOUT.
**Pre-registered success criterion: holdout PF ≥ 1.3 after 0.4% cost, CI95% lower bound > 1.0, positive (or non-negative) in ≥2 regimes, survives the crash jackknife.** Otherwise: no deployable edge, honest conclusion.

## Tested hypothesis log (anti-data-dredging)
| # | Hypothesis | Set | Result | CI95% (det−rnd diff) |
|---|---|---|---|---|
| H1 | Detector beats matched random (4h/150d) | 4h/150d, 941 det vs 4347 rnd | **REJECTED ex-crash**: GLOBAL & BULL & BEAR = chance, RANGE worse than random. Raw edge = Feb crash only | GLOBAL ex-crash [−0.94,+0.36]; BEAR ex-crash [−1.19,+0.95]; RANGE [−2.70,−0.52] |

| H2 | Cross-sectional momentum beats B&H and random (1d, weekly rebal) | 25 symbols, 300d daily, 34-40 rebal | **INCONCLUSIVE (underpowered)**: no lookback/variant beats B&H or random (diff CI ⊃ 0); long-only crushed by alt bear (B&H −54%); L/S ≈ random | L=30 L/S vs B&H [−3.3,+3.6]%; OOS [−1.4,+6.1]% |

## GATE PHASE 1 = FAIL (patterns without edge). PHASE 3bis = INCONCLUSIVE (momentum, but underpowered).
## DATA FIX FOUND: **Binance serves 4 years of daily** (Bitget caps at ~200 deep bars). `scripts/momentum_test.py`/`trend_test.py` pull from Binance.

| H2b | Cross-sectional momentum over 4 YEARS (Binance, 23 sym, ~200 rebal, multi-cycle) | clean/powered | **REJECTED**: L/S (isolates alpha) = flat/negative; long-only ≈ beta (does not beat B&H, CI ⊃ 0). No momentum alpha | L=30 L/S vs B&H [−2.4,+0.8]%; OOS no edge |
| H3 | Trend-following (TSMOM long/cash) beats B&H on risk (4 years, 22 sym) | clean/powered | **PARTIAL**: no RETURN alpha (diff CI ⊃ 0) BUT consistently reduces DRAWDOWN (−68% → −42/53% maxDD, Sharpe 0.23 → 0.36-0.38) and holds OOS (2 halves). = risk management, not prediction | diff ret/d [−0.21,+0.21] (non-sig); DD/Sharpe robust |

| H4 | MASTER setup "liquidity sweep + CHoCH reversal" (SMC) beats matched random | Binance 4h, 15 sym, 500d multi-regime, 571 setups; RR×volume grid | **REJECTED**: 6 variants (RR 1.5/2/3 × vol off/>1.5×), NONE beats random (GLOBAL = chance), several WORSE ex-crash. WR sticks to each RR's break-even = coin-flip direction. Volume filter does not help | global all ⊃ 0; ex-crash anti or ⊃ 0 |

| H5 | Detection QUALITY / DISCRIMINATION (geom_confidence, confluence_score) separates winners | 4h/150d dataset, 947 trades | **REJECTED**: in-sample quality seems to discriminate (Q4 EDGE) but it is a directional artefact (high quality ≈ shorts in bear). OOS (2nd half): ALL negative and HIGH quality is WORSE (geom Q4 −1.92%, Q3 −3.27%). At fixed direction: no monotonic gradient. Tightening the sort does not help | OOS quartiles all E[R]<0 |

| H6 | INDEPENDENT variable (funding rate) discriminates / improves PF | Binance perp, 12 sym, 2 years, 8748 obs (`scripts`/`data/ml/funding_panel.csv`) | **REJECTED**: Test 1 (fwd by funding quintile) non-monotonic, no contrarian pattern = noise. Test 2 (cross-sectional long-low/short-high) gross Sharpe 1.16 BUT CI ⊃ 0 (non-sig) and NET after 0.1%/d fees = −0.008%/d (negative). Binance OI too short (~30d) to test | gross [−0.016,+0.204]; net negative |

## PROGRAM CONCLUSION: no PREDICTION/RETURN edge (broad patterns, ML, X-sectional momentum, sweep+CHoCH+volume, quality/discrimination sort, multi-TF Bollinger, AND the funding independent variable — all rejected vs random/OOS/costs). Independent variables NOT testable with our data (L2 order-flow, paid on-chain, external macro) = the only remaining frontier, but out of reach without new sources. The ONLY robust result = **trend-following reduces drawdown** (crisis-alpha, risk management). Realistic objective = trend-filtered exposure + risk control, NOT "predict/beat the market". A DISCRETIONARY edge (non-mechanizable human judgement) can only be validated in **paper-forward** (log real trades and measure them vs random), not in backtest.
