# Exoplanet vetting status
Updated: 2026-10-21, DESKTOP-BBASC81 (last update from the desktop)

## Headline
**Moving to the user's 8 GB laptop (10/21);** a new laptop session takes over, starting from `HANDOFF.md`. No new
jobs run on the desktop. Phase 1 is finished and tested once. Phase 2 (`PHASE2_PLAN.md`): M0 pilot and M1 download
are done, and every star's light curves are cached. Next is M2 (views + QA for all KOIs).

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

Phase 2 M1 download, finished 10/20 16:50: **7,927 / 7,927 stars cached, 0 errors** (gate ≥ 98% of KOIs: passed,
100%). 53 GB raw streamed into a 14.3 GB cache. Median 17 quarters per star; 160 stars have < 4 quarters (QA
group for M2).

## Running now
Nothing (and nothing new will start on the desktop).

## Waiting on user
- Laptop setup: copy `data/lightcurves/` (14.3 GB cache + manifest + pilot views) from the desktop (recommended), or
  re-run the download there (~2.3 h, resumable). Setup steps are in `HANDOFF.md`.

## Next up
1. Laptop: setup + smoke test (`HANDOFF.md` §2), light-curve cache in place.
2. M2: detrend + views + QA for all KOIs (average view per class on the pool, TTV and no-transit QA flags).
3. M3: Phase 2a light-curve features with the existing CV protocol, compared with Phase 1.
