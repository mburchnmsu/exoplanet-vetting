# Exoplanet vetting status
Updated: 2026-10-18 22:03, DESKTOP-BBASC81

## Headline
Phase 1 (catalog-feature classifier) is finished and tested once on the sealed test set, which confirmed the
cross-validation numbers. Phase 2 (adding Kepler light curves) is planned in `PHASE2_PLAN.md` and waiting on the user's
decisions.

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

## Running now
Nothing.

## Waiting on user
- Phase 2 decisions D1–D7 in `PHASE2_PLAN.md`: which machine, raw vs cached light-curve storage, features-first vs
  CNN-first, centroid time series, reusing the test split once, CNN CV budget, data kept out of git. Each has a
  recommendation.

## Next up
1. Phase 2, M0 pilot: download ~50 stars' light curves, build the cache, detrending and folded views, then measure
   the real download size and time.
2. M1: full overnight light-curve download at low priority.
