# Exoplanet vetting status
Updated: 2026-10-24 01:40, DESKTOP-BBASC81 (Thinkpad)

## Headline
**Exoplanet is back on the Thinkpad (10/22), and this session is in charge again;** the Intel laptop setup was
abandoned (precompile stopped for low memory three times). Phase 1 is finished and tested once. Phase 2 M0 + M1 are
done: the local light-curve cache is intact (7,927 stars, 14.3 GB). M2 views + QA are built for every KOI; the M2
QA notebook (average views per class, pool only) is next.

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

Phase 2 M2 build, finished 10/24 03:10 (1.4 h, alongside Zodi): **views + QA for 9,271 KOIs on all 7,927 stars, 0
errors** (one star with nonsense catalog ephemerides, duration > period, needed a masking fallback). Views:
`data/views_all.jld2` (119 MB, git-ignored, rebuildable in ~1.5 h). QA table: `sweeps/results/p2_qa.csv`. QA flags:
20 KOIs with < 2 observed transits, 291 with measured depth > 2× off the catalog. The timing-scatter flag is too
loose as built (3,095); it is being refined in the M2 QA notebook.

## Running now
Nothing.

## Waiting on user
Nothing.

## Next up
On the Thinkpad (cache at `C:\JuliaProjects\Exoplanet vetting\data\lightcurves`, M1 already done):
1. M2 QA notebook (light): average views per class on the pool, QA flags; check the M2 gate (EB average shows a
   secondary, planet average U-shaped, no systematic artifacts).
2. M3: Phase 2a light-curve features with the existing CV protocol (needs the user's go; light-to-moderate CPU).
