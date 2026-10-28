# Exoplanet vetting status
Updated: 2026-10-28 16:25, DESKTOP-BBASC81 (Thinkpad)

## Headline
**M3 (Phase 2a) done: our own light-curve features clearly improve both families** (CV, DR25: catalog 0.955 → 0.971,
now level with the robovetter's `koi_score` at 0.970; + centroid 0.983 → 0.987). Phase 1 is finished and tested once;
Phase 2 M0–M3 are done. A public-facing `README.md` is at the repo root.

## Best model
Frozen (Phase 1, `models/step5_frozen.jld2`): two 50/50 averages of a rank-gauss-input MLP (64-32, 10 seeds) and
EvoTrees (depth 8, 10 seeds). Headline operating point: 98% planet recall.
- **Catalog-only**: sealed-test DR25 ROC-AUC 0.957 [0.944, 0.970]; at 98% recall precision 0.77, 31% of FPs kept.
- **Catalog + centroid** (partly circular with Kepler's centroid test): 0.988 [0.982, 0.994]; precision 0.87, 16% kept.

Best in CV (not yet frozen): the same recipes + the 11 Phase 2a light-curve features (+ 2 of our centroid features in
the "+ centroid" family): DR25 CV ROC-AUC **0.971** / **0.987**.

## Latest results
M3, CV only (standard folds, 5 shuffles × 10 seeds; paired star-bootstrap 95% CI; `sweeps/results/p2_cv_summary.csv`):

| MLP + trees average | Subset | Phase 1 | + LC features | Δ [95% CI] |
|---|---|---|---|---|
| Catalog | DR25 (7,199) | 0.955 | **0.971** | +0.015 [+0.012, +0.018] |
| Catalog | supplemental (1,089) | 0.668 | 0.696 | +0.028 [+0.004, +0.055] |
| Catalog + centroid | DR25 | 0.983 | **0.987** | +0.004 [+0.003, +0.005] |
| Catalog + centroid | supplemental | 0.755 | 0.766 | +0.011 [−0.008, +0.031] |

Single learners, DR25: trees +0.017 / +0.005, MLP +0.015 / +0.004, all CIs above 0. In the + centroid family the
average equals the trees alone (0.987).

Features: 9,271 KOIs, 0 errors, ~1 h (`sweeps/results/p2_lcfeatures.csv`); CV took ~33 min for all four runs.
Sealed test (Phase 1, scored once on 10/13): catalog-only 0.957, + centroid 0.988, `koi_score` 0.979.

Earlier Phase 2: M0 pilot 49/49 stars; M1 7,927/7,927 stars cached (14.3 GB); M2 views + QA for 9,271 KOIs, gate
passed (notebooks `exo101926`, `exo102426`).

## Running now
**M4 (Phase 2b CNNs), started 10/28 ~16:25 on the user's go** (Zodi idle; below-normal priority, 2 threads, ~2.4 GB):
`sweeps/p2_cnn.jl --auto`, log `sweeps/results/p2_cnn.log`, resumable per variant.
- Timing gate passed (`p2_cnn_gate.log`): ~182k-parameter two-tower CNN, 13–16 s per epoch, one full model 4.3 min
  (best epoch 11). On repeat 1 / fold 1, one CNN alone is below Phase 2a (DR25 0.959 vs 0.968), so the question is
  whether it adds in an ensemble.
- Screens (D6: 1 fold shuffle × 2 seeds) of 5 variants: views only; + catalog; + catalog + LC features; centroid
  channels + catalog + centroid; + all. ~3.8 h.
- Then the final 5 × 3 CV for the best variant of each family (ranked by DR25 AUC of its 50/50 with Phase 2a):
  ~5.5 h each → done ~10/29 morning (~15 h total).

## Waiting on user
- Public repo (scan sent to the hub): commit author email in history; whether to keep CLAUDE.md / HANDOFF.md /
  STATUS.md public. No secrets found.
- Optional: back up `data/lightcurves/cache/` (14.3 GB, git-ignored; ~2.5 h to re-download).

## Next up
1. M3 notebook: feature distributions per class (pool), feature importance, blind-spot table (by FP flag, period,
   SNR, QA flags) and where Phase 2a fixes / breaks Phase 1 calls.
2. M4 results: M4 gate = beats 2a, or adds to it in an ensemble (CI excludes 0). Then M5 (ensemble, calibration,
   freeze) and M6 (the one Phase 2 test scoring).
