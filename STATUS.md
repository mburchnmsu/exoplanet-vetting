# Exoplanet vetting status
Updated: 2026-10-30, DESKTOP-BBASC81 (Thinkpad)

## Headline
**Phase 2 is done (M0–M6). On the sealed test set, our light-curve features lift DR25 ROC-AUC 0.957 → 0.976
(catalog-only, near the robovetter's `koi_score` 0.979) and 0.988 → 0.993 (+ centroid)**, both paired gains with
CIs above 0, matching CV. The model is Phase 2a (MLP + trees with our LC features); the CNNs (M4) neither beat nor
added to it. The repo is public; `README.md` shows the Phase 1 and Phase 2 test results.

## Best model
Frozen Phase 2 (`models/p2_frozen.jld2`): two 50/50 averages of a rank-gauss-input MLP (64-32, 10 seeds) and
EvoTrees (depth 8, 10 seeds), on the catalog features + our 11 light-curve features (+ DV centroid + our 2
centroid features in the "+ centroid" family). No calibrator. Headline operating point: 98% planet recall.
- **Catalog-only**: sealed-test DR25 ROC-AUC **0.976** [0.966, 0.984]; at 98% recall precision 0.82, 23% of FPs kept.
- **Catalog + centroid** (partly circular with Kepler's centroid test): **0.993** [0.988, 0.996]; precision 0.90,
  12% kept.

Phase 1 (`models/step5_frozen.jld2`, catalog only): test 0.957 / 0.988.

## Latest results
**M6, the one Phase 2 sealed-test scoring (10/30, `sweeps/p2_test.jl`, user's go-ahead; same 913 test KOIs /
792 stars as Phase 1; `sweeps/results/p2_test_*.csv`). The test set confirms CV: Phase 2a beats Phase 1 on the
test set too, in both families.**

| DR25 KOIs (796) | Test ROC-AUC [95% CI] | CV expected | Phase 1 test | Δ vs Phase 1, paired [95% CI] |
|---|---|---|---|---|
| Catalog-only | **0.976** [0.966, 0.984] | 0.970 | 0.957 | +0.018 [+0.011, +0.027] |
| Catalog + centroid | **0.993** [0.988, 0.996] | 0.987 | 0.988 | +0.004 [+0.001, +0.009] |
| Robovetter `koi_score` (reference) | 0.979 | — | — | — |

- Supplemental KOIs (117): 0.760 / 0.797 (Phase 1 0.724 / 0.788; Δ CIs include 0, small n).
- **98% planet recall (frozen thresholds), DR25 test:** catalog-only recall 0.985 [0.972, 0.995], precision 0.815,
  23% of FPs kept (Phase 1 test: 31%; CV 25%); + centroid recall 0.990 [0.979, 0.998], precision 0.897, 12% kept
  (Phase 1 test: 16%; CV 14%). CONFIRMED planets lost: 0.4% / 0%.
- Every CV expectation lies inside the test CI. Like Phase 1, the test split scores a little higher than CV.

**M5 analysis (10/29, from the saved CV; `sweeps/p2_m5.jl --analyze`, `sweeps/results/p2_m5_*.csv`).** Phase 1
rows reproduce step 5's menu exactly.
- **Calibration: still not needed.** Platt fit a = 1.035 / 1.038, b ≈ 0; nested Platt/isotonic don't improve DR25
  log-loss (0.223 / 0.147 uncalibrated). DR25 ECE 0.012. Supplemental KOIs remain miscalibrated (ECE 0.05–0.06).
- **98% planet recall (headline), nested, DR25:**

| Family | Phase | Threshold | Precision | FPs kept | FPs kept /1000 | CONFIRMED lost |
|---|---|---|---|---|---|---|
| Catalog | Phase 1 | 0.15 | 0.745 | 35% | 172 | 0.4% |
| Catalog | **Phase 2a** | 0.18 | **0.805** | **25%** | 122 | 0.4% |
| + centroid | Phase 1 | 0.21 | 0.856 | 17% | 84 | 0.4% |
| + centroid | **Phase 2a** | 0.25 | **0.882** | **14%** | 67 | 0.3% |

- **Plan §1 criteria:**
  1. Main (beats Phase 1, CI > 0): **met** in both families (M3).
  2. Blind spots: **partly.** At p = 0.5, NT-only FPs missed 22.8 → 20.8% (catalog) / 27.0 → 23.7% (+ centroid);
     EC-only 21.4 → 18.0% / 26.3 → 23.2%. Both fall, but not below the 15% target. Supplemental AUC rises
     (0.664 → 0.692 / 0.751 → 0.762). USP planets (P < 1.5 d) called FP: 59.7 → 40.2% / 32.0 → 26.5%.
  3. No regressions: **met.** CONFIRMED lost at 98% recall 0.4 → 0.4% / 0.4 → 0.3%.
  4. Stretch (catalog-only reaches `koi_score`'s 0.970): **met** (0.970).

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
Nothing. **M5 freeze done 10/29 20:54** (took 2 min): `models/p2_frozen.jld2` (44 MB) holds both Phase 2a models
(10 MLP + 10 tree seeds each, whole pool, inner split seed 5005), their preprocessing, the menu thresholds (98%
recall: catalog 0.185, + centroid 0.244) and calibrator "none". Round trip OK (in-sample pool AUC 0.980 / 0.994).
Log `sweeps/results/p2_m5_freeze.log`. The test split was not touched.

M4 was stopped 10/29 ~20:30 (user's decision, during the cent_lc final; nothing lost that matters).
M4 run notes: CNNs need ~1 h per screen and ~6.4 h per 5 × 3 final at 2 threads, ~2 GB; launched detached
outside Claude Code after Claude Code's memory safeguard killed two earlier attempts.

## Waiting on user
- What comes next: the Phase 2 write-up notebook, and any further work (e.g. the NT/EC blind spots).
- Public repo: the user OK'd going public as-is (10/30; email in history and notes files stay). Final pass found
  nothing sensitive. The user flips visibility themselves.
- Optional: back up `data/lightcurves/cache/` (14.3 GB, git-ignored; ~2.5 h to re-download).

## Next up
1. Phase 2 write-up notebook (M3–M6: feature distributions, blind-spot table, CNN result, menu, test results).
2. README: updated with the Phase 2 results and `figures/p2_test_roc.png` (10/30, cff12aa; repo is public).
