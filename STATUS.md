# Exoplanet vetting status
Updated: 2026-10-29 20:45, DESKTOP-BBASC81 (Thinkpad)

## Headline
**M4 (CNN) closed: the CNNs neither beat Phase 2a nor add to it, so Phase 2a is the Phase 2 model.** Starting M5
(calibration, operating-point menu, success criteria, freeze). Phase 2a's own light-curve features lift DR25 CV
ROC-AUC 0.955 → 0.971 (catalog, level with the robovetter's `koi_score` 0.970) and 0.983 → 0.987 (+ centroid).
Phase 1 is finished and tested once; Phase 2 M0–M4 are done. A public-facing `README.md` is at the repo root.

## Best model
Frozen (Phase 1, `models/step5_frozen.jld2`): two 50/50 averages of a rank-gauss-input MLP (64-32, 10 seeds) and
EvoTrees (depth 8, 10 seeds). Headline operating point: 98% planet recall.
- **Catalog-only**: sealed-test DR25 ROC-AUC 0.957 [0.944, 0.970]; at 98% recall precision 0.77, 31% of FPs kept.
- **Catalog + centroid** (partly circular with Kepler's centroid test): 0.988 [0.982, 0.994]; precision 0.87, 16% kept.

Best in CV, to be frozen in M5: the same recipes + the 11 Phase 2a light-curve features (+ 2 of our centroid
features in the "+ centroid" family): DR25 CV ROC-AUC **0.971** / **0.987**.

## Latest results
**M4 (10/29; paired star bootstrap vs the Phase 2a MLP + trees average, DR25): no CNN variant beats Phase 2a, and
none adds to it in a 50/50 ensemble. Gate failed; stopped by the user's decision before the cent_lc final.**

Screens (1 fold shuffle × 2 seeds):

| Variant | CNN alone | Phase 2a | Δ [95% CI] | 50/50 with 2a: Δ [95% CI] |
|---|---|---|---|---|
| views only | 0.949 | 0.969 | −0.020 [−0.024, −0.017] | −0.002 [−0.004, −0.001] |
| + catalog | 0.958 | 0.969 | −0.011 [−0.014, −0.009] | −0.001 [−0.002, +0.000] |
| + catalog + LC | 0.962 | 0.969 | −0.007 [−0.009, −0.005] | −0.001 [−0.002, −0.000] |
| centroid + catalog + centroid | 0.980 | 0.987 | −0.007 [−0.009, −0.005] | −0.001 [−0.002, −0.000] |
| centroid + all | 0.983 | 0.987 | −0.004 [−0.005, −0.003] | −0.001 [−0.001, −0.000] |

**Final 5 × 3 CV, hybrid_lc (6.4 h) confirms it**: DR25 CNN 0.964 vs Phase 2a 0.971, Δ −0.007 [−0.008, −0.005];
50/50 with 2a 0.969, Δ −0.001 [−0.002, −0.000]. All KOIs: CNN 0.946 vs 0.954; 50/50 Δ −0.001 [−0.002, −0.001].
All-KOI screen results show the same pattern (`sweeps/results/p2_cnn.log`). The views alone carry most of the
signal (0.949), but the tabular models with our LC features already capture it.

M3, CV only (standard folds, 5 shuffles × 10 seeds; paired star-bootstrap 95% CI; `sweeps/results/p2_cv_summary.csv`):

| MLP + trees average | Subset | Phase 1 | + LC features | Δ [95% CI] |
|---|---|---|---|---|
| Catalog | DR25 (7,199) | 0.955 | **0.971** | +0.015 [+0.012, +0.018] |
| Catalog | supplemental (1,089) | 0.668 | 0.696 | +0.028 [+0.004, +0.055] |
| Catalog + centroid | DR25 | 0.983 | **0.987** | +0.004 [+0.003, +0.005] |
| Catalog + centroid | supplemental | 0.755 | 0.766 | +0.011 [−0.008, +0.031] |

Single learners, DR25: trees +0.017 / +0.005, MLP +0.015 / +0.004, all CIs above 0. In the + centroid family the
average equals the trees alone (0.987).

Sealed test (Phase 1, scored once on 10/13): catalog-only 0.957, + centroid 0.988, `koi_score` 0.979.

Earlier Phase 2: M0 pilot 49/49 stars; M1 7,927/7,927 stars cached (14.3 GB); M2 views + QA for 9,271 KOIs, gate
passed (notebooks `exo101926`, `exo102426`); M3 features 9,271 KOIs, 0 errors.

## Running now
Nothing. M4 was stopped 10/29 ~20:30 (user's decision, during the cent_lc final; nothing lost that matters).
M4 run notes: CNNs need ~1 h per screen and ~6.4 h per 5 × 3 final at 2 threads, ~2 GB; launched detached
outside Claude Code after Claude Code's memory safeguard killed two earlier attempts.

## Waiting on user
- Public repo (scan sent to the hub): commit author email in history; whether to keep CLAUDE.md / HANDOFF.md /
  STATUS.md public. No secrets found.
- Optional: back up `data/lightcurves/cache/` (14.3 GB, git-ignored; ~2.5 h to re-download).

## Next up
1. M5 (`sweeps/p2_m5.jl`): (a) from the saved CV, light: nested calibration check, operating-point menu, and the
   plan's §1 criteria (NT/EC miss rates, supplemental AUC, USP planets, CONFIRMED lost at 98% recall), Phase 1 vs
   Phase 2a; (b) freeze both Phase 2a models on the whole pool (`models/p2_frozen.jld2`), ~2 GB, 2 threads.
2. M6: the one Phase 2 sealed-test scoring (D5), with the user's go-ahead.
3. Notebook for M3–M5 (feature distributions, blind-spot table, menu).
