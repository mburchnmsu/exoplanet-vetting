# Exoplanet vetting status
Updated: 2026-10-29 20:23, DESKTOP-BBASC81 (Thinkpad)

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
**M4 screens (10/29; 1 fold shuffle × 2 seeds, paired star bootstrap vs the Phase 2a MLP + trees average; DR25):
no CNN variant beats Phase 2a, and none adds to it in a 50/50 ensemble.**

| Variant | CNN alone | Phase 2a | Δ [95% CI] | 50/50 with 2a: Δ [95% CI] |
|---|---|---|---|---|
| views only | 0.949 | 0.969 | −0.020 [−0.024, −0.017] | −0.002 [−0.004, −0.001] |
| + catalog | 0.958 | 0.969 | −0.011 [−0.014, −0.009] | −0.001 [−0.002, +0.000] |
| + catalog + LC | 0.962 | 0.969 | −0.007 [−0.009, −0.005] | −0.001 [−0.002, −0.000] |
| centroid + catalog + centroid | 0.980 | 0.987 | −0.007 [−0.009, −0.005] | −0.001 [−0.002, −0.000] |
| centroid + all | 0.983 | 0.987 | −0.004 [−0.005, −0.003] | −0.001 [−0.001, −0.000] |

**Final 5 × 3 CV, hybrid_lc (10/29 19:47, 6.4 h) confirms it**: DR25 CNN 0.964 vs Phase 2a 0.971, Δ −0.007
[−0.008, −0.005]; 50/50 with 2a 0.969, Δ −0.001 [−0.002, −0.000]. All KOIs: CNN 0.946 vs 0.954; 50/50 Δ −0.001
[−0.002, −0.001].

All-KOI results show the same pattern (`sweeps/results/p2_cnn.log`). The views alone carry most of the signal
(0.949), but the tabular models already capture it. Family picks for the finals: hybrid_lc and (next) cent_lc.

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
**M4 resumed 10/29 09:38, launched detached outside Claude Code** (julia PID 33652, BelowNormal, 2 threads; user's
choice, alongside Zodi's daytime test) → ~23 h, done ~10/30 morning. Log appends to `sweeps/results/p2_cnn.log`;
no automatic end notice, so the log is checked periodically. Stop it with `Stop-Process -Id 33652`.
- Saved so far: views screen (10/28, 1.3 h), hybrid screen (10/29 10:43, 1.0 h, median best epoch 12),
  hybrid_lc screen (11:37, 0.9 h, median best epoch 10), cent screen (12:35, 1.0 h, median best epoch 10),
  cent_lc screen (13:25, 0.8 h, median best epoch 9). hybrid_lc final (5 × 3) saved 19:47 (6.4 h). **Now the
  cent_lc final since 19:47, ~6.5 h** → done ~10/30 02:30. The screens already fail the M4 gate, so whether to let the finals
  run is a question for the user. AUC comparisons are printed once all five screens are done.
- History: **M4 stopped ~22:00 on 10/28**: Claude Code's memory reaper killed it (system low on memory: 5.9 GB
  free afterwards, with two other Julia processes at 3.5 GB and 1.7 GB).
- Saved before the stop: the **views-only screen** (`cv_cnn_views_r1s2.jld2`, 1.3 h, median best epoch 11). The
  hybrid screen had just started; the run resumes per variant, so this run skips views.
- Timing is slower than the gate estimate: 1.3 h per screen (not ~45 min), so the rest is ~5 h of screens + ~2 ×
  9 h of finals (~23 h) at 2 threads.
- The first attempt (10/28 16:23) also died after 8 min, most likely the same reaper.
- To stop the reaper killing it: start Claude Code with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`, or launch
  the job detached outside Claude Code, or run when other Julia jobs are closed.

M4 plan, as launched (below-normal priority, 2 threads, ~2.4 GB): `sweeps/p2_cnn.jl --auto`, log
`sweeps/results/p2_cnn.log`, resumable per variant.
- Timing gate passed (`p2_cnn_gate.log`): ~182k-parameter two-tower CNN, 13–16 s per epoch, one full model 4.3 min
  (best epoch 11). On repeat 1 / fold 1, one CNN alone is below Phase 2a (DR25 0.959 vs 0.968), so the question is
  whether it adds in an ensemble.
- Screens (D6: 1 fold shuffle × 2 seeds) of 5 variants: views only; + catalog; + catalog + LC features; centroid
  channels + catalog + centroid; + all. ~3.8 h.
- Then the final 5 × 3 CV for the best variant of each family (ranked by DR25 AUC of its 50/50 with Phase 2a):
  ~5.5 h each → done ~10/29 morning (~15 h total).

## Waiting on user
- M4: let the cent_lc final run (~6.5 h more, definitive 5 × 3 numbers for the centroid family), or stop now and go
  to M5 with Phase 2a as the model?
- Public repo (scan sent to the hub): commit author email in history; whether to keep CLAUDE.md / HANDOFF.md /
  STATUS.md public. No secrets found.
- Optional: back up `data/lightcurves/cache/` (14.3 GB, git-ignored; ~2.5 h to re-download).

## Next up
1. M3 notebook: feature distributions per class (pool), feature importance, blind-spot table (by FP flag, period,
   SNR, QA flags) and where Phase 2a fixes / breaks Phase 1 calls.
2. M4 results: M4 gate = beats 2a, or adds to it in an ensemble (CI excludes 0). Then M5 (ensemble, calibration,
   freeze) and M6 (the one Phase 2 test scoring).
