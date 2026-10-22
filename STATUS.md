# Exoplanet vetting status
Updated: 2026-10-21 20:45, Intel (8 GB laptop)

## Headline
**Setup on the Intel laptop (8 GB) in progress (10/21).** Repo cloned; Julia 1.12.7 pinned to the repo folder with
`juliaup override` (Manifest resolved on 1.12.6; laptop default is 1.13). Precompile is ~351 packages in, with the heavy
ones (Flux, CairoMakie) left: Claude Code stopped it twice for low memory. Smoke test not run yet. Phase 1 finished and
tested once; Phase 2 M0 + M1 done on the desktop, and the laptop re-downloads M1 (no cache copied).

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
Nothing on the Intel laptop. Safe to restart.

## Waiting on user
- Finish the precompile outside Claude Code's low-memory killer: either run the smoke test in a plain PowerShell window
  with `$env:JULIA_NUM_PRECOMPILE_TASKS = 1` (`HANDOFF.md` §2 step 4; should print `(9201, 15)`), or start Claude Code
  with `$env:CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP = 1` set beforehand.

## Next up
Resume after a restart:
1. Start Claude Code with `$env:CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP = 1` set first (needed for the long runs).
2. In the repo, `julia --version` should say 1.12.7 (the juliaup override persists). Finish the precompile + smoke test
   with `JULIA_NUM_PRECOMPILE_TASKS=1`; packages compiled so far are kept in `~/.julia/compiled/v1.12`.
3. Push the setup milestone, then start M1: `julia --project=. -t 2 sweeps/p2_download.jl` (all 7,927 stars, ~2.3 h,
   resumable, < 1 GB RAM). Tell the hub when it starts and finishes.
4. M2: detrend + views + QA for all KOIs. 5. M3: Phase 2a light-curve features with the existing CV protocol.
