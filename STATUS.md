# Exoplanet vetting status
Updated: 2026-10-26, DESKTOP-BBASC81 (Thinkpad)

## Headline
**Exoplanet is back on the Thinkpad (10/22), and this session is in charge again;** the Intel laptop setup was
abandoned (precompile stopped for low memory three times). Phase 1 is finished and tested once. Phase 2 M0 + M1 are
done: the local light-curve cache is intact (7,927 stars, 14.3 GB). M2 views + QA are built for every KOI and passed
the gate. M3 (Phase 2a light-curve features + CV) is running since 10/26.

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
20 KOIs with < 2 observed transits, 291 (3.1%) with depth > 2× off the catalog, 222 (2.4%) with large transit-timing
scatter (catches Kepler-9 b). **M2 gate passed** (notebook `exo102426.ipynb`): planets U-shaped, EBs V-shaped, no
artifacts; secondary and odd/even signals are noise-dominated in class medians, so M3 features must be significances.

## Running now
**M3 (Phase 2a), started 10/26 on the user's go**, alongside Zodi round 34 (below-normal priority, 2 threads, ~2 GB):
1. `sweeps/p2_features.jl`: 11 light-curve features (`src/LCFeatures.jl`; significances against box noise: SNR,
   trapezoid depth ratio / V-shape / duration ratio, odd-even σ, secondary σ / ratio / phase, single-event ratio,
   timing scatter, observed transits) + 2 centroid features for the "+ centroid" family. Test: 0.78 s/star → ~1.7 h.
   Resumable shards in `data/lightcurves/features/`; output `sweeps/results/p2_lcfeatures.csv`.
2. Then `sweeps/p2_cv.jl`: trees + quantile MLP, both families, 5 shuffles × 10 seeds, paired vs Phase 1
   (est. ~1–2 h per family). Logs: `sweeps/results/p2_features.log`, `p2_cv.log`.

## Waiting on user
- M4 (CNN) timing: when the CPU-heavy timing gate and screens may run (suggested: after Zodi round 34 ends).
- Optional: back up `data/lightcurves/cache/` (14.3 GB, git-ignored; ~2.5 h to re-download).

## Next up
1. M3 results: QA notebook (feature distributions per class on the pool, paired Δ vs Phase 1 on DR25, blind-spot
   table by FP flag / period / SNR / QA flags).
2. M4: CNN timing gate and screens.
