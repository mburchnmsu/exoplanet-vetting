# Exoplanet vetting status
Updated: 2026-10-19 12:43, DESKTOP-BBASC81

## Headline
Phase 1 (catalog-feature classifier) is finished and tested once on the sealed test set, which confirmed the
cross-validation numbers. Phase 2 (adding Kepler light curves, `PHASE2_PLAN.md`) is approved and its M0 pilot is
running.

## Best model
Two frozen models (`models/step5_frozen.jld2`). Each is a 50/50 average of a rank-gauss-input MLP (64-32, 10 seeds) and
gradient-boosted trees (EvoTrees depth 8, 10 seeds). No calibrator (already calibrated). Headline operating point: 98%
planet recall.
- **Catalog-only** (11 catalog features): sealed-test DR25 ROC-AUC 0.957 [0.944, 0.970]. At 98% recall: precision 0.77,
  31% of false positives kept.
- **Catalog + centroid** (adds 7 centroid features; partly circular with Kepler's own centroid test): ROC-AUC 0.988
  [0.982, 0.994]. At 98% recall: precision 0.87, 16% of false positives kept.

## Latest results
Sealed test, scored once on 10/13 (796 DR25 KOIs; star-bootstrap 95% CI), next to 5-fold CV (5 shuffles Ã— 10 seeds):

| Model | Test ROC-AUC | CV ROC-AUC | Test recall / precision at the 98% threshold |
|---|---|---|---|
| Catalog-only | 0.957 | 0.954 | 0.988 / 0.771 |
| Catalog + centroid | 0.988 | 0.983 | 0.993 / 0.869 |
| Kepler `koi_score` (reference) | 0.979 | â€” | â€” |

Post-test experiment (CV only): training on CONFIRMED vs FP only is worse at the task (âˆ’0.015 / âˆ’0.007 AUC). Not
adopted.

## Running now
Phase 2 M0 pilot (`sweeps/p2_pilot.jl`, started 10/19, low priority, 2 threads): downloading 49 pool stars' Kepler
light curves from MAST, building the compact cache, detrending and folded views, and timing each step. Expected to
take well under an hour.

## Waiting on user
Nothing. D1â€“D7 approved 10/19 (all recommendations).

## Next up
1. M0: notebook of raw vs detrended light curves and views for example KOIs; check the gate (pipeline works, views
   look right, size/time estimate).
2. M1: full light-curve download of all 7,927 stars, overnight at low priority, resumable.
3. M2: views + QA for all KOIs; M3: Phase 2a light-curve features with the existing CV protocol.
