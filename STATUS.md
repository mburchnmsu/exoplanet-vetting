# Exoplanet vetting status
Updated: 2026-10-19 13:05, DESKTOP-BBASC81

## Headline
Phase 1 (catalog-feature classifier) is finished and tested once on the sealed test set, which confirmed the
cross-validation numbers. Phase 2 (adding Kepler light curves, `PHASE2_PLAN.md`): the M0 pilot passed its gate, and
the full light-curve download (M1) is running.

## Best model
Two frozen models (`models/step5_frozen.jld2`). Each is a 50/50 average of a rank-gauss-input MLP (64-32, 10 seeds) and
gradient-boosted trees (EvoTrees depth 8, 10 seeds). No calibrator (already calibrated). Headline operating point: 98%
planet recall.
- **Catalog-only** (11 catalog features): sealed-test DR25 ROC-AUC 0.957 [0.944, 0.970]. At 98% recall: precision 0.77,
  31% of false positives kept.
- **Catalog + centroid** (adds 7 centroid features; partly circular with Kepler's own centroid test): ROC-AUC 0.988
  [0.982, 0.994]. At 98% recall: precision 0.87, 16% of false positives kept.

## Latest results
Sealed test, scored once on 10/13 (796 DR25 KOIs; star-bootstrap 95% CI), next to 5-fold CV (5 shuffles × 10 seeds):

| Model | Test ROC-AUC | CV ROC-AUC | Test recall / precision at the 98% threshold |
|---|---|---|---|
| Catalog-only | 0.957 | 0.954 | 0.988 / 0.771 |
| Catalog + centroid | 0.988 | 0.983 | 0.993 / 0.869 |
| Kepler `koi_score` (reference) | 0.979 | — | — |

Post-test experiment (CV only): training on CONFIRMED vs FP only is worse at the task (−0.015 / −0.007 AUC). Not
adopted.

Phase 2 M0 pilot (10/19, 49 pool stars, 71 KOIs, notebook `exo101926.ipynb`): 49/49 downloaded, no errors. Measured
~1 s download + ~1.4 s detrend/views per star → full set ~2.3 h download, 55 GB raw streamed to a ~15 GB cache.
Views look right (EBs V-shaped, planets U-shaped; centroid shift ~0.05 mpix planets vs ~0.8 mpix CO/SS FPs).

## Running now
Nothing. Phase 2 M1 (`sweeps/p2_download.jl`) restarted at 21:06 but was stopped again by Claude Code within minutes
because the PC ran low on memory: two other Julia jobs (~2.5 GB each) started at about the same time, and memory
compression was at ~6 GB. Not an error. 524 of 7,927 stars are cached, 0 errors; it resumes from there.
Per the user (10/19 evening): Zodi is the only project running tonight. M1 stays paused at 524/7,927; no restart
checks are scheduled.

## Waiting on user
- When to restart M1 (~2.2 h left; resumable). Note: Claude Code stops its own background jobs whenever the whole
  PC is low on memory, so a restart works best when the other projects are idle, or launched from a normal terminal.

## Next up
1. M1 check: ≥ 98% of KOIs have data; list failures.
2. M2: detrend + views + QA for all KOIs (average view per class on the pool, TTV and no-transit QA flags).
3. M3: Phase 2a light-curve features with the existing CV protocol, compared with Phase 1.
