# Kepler vetting: noise floor and error analysis of the baseline MLP

*10/10/2026 · notebook `exo101026.ipynb` · compute `sweeps/cv_noise_floor.jl` · results in `sweeps/results/`*

## Setup

- **Data:** Kepler cumulative KOI table. 9,201 KOIs have all 11 features. Labels: CONFIRMED + CANDIDATE = 1, FALSE POSITIVE = 0.
- **Baseline model (from `exo100926`):**
  - 11 features (period, duration, depth, impact, model SNR, prad, teq, steff, srad, slogg, koi_count)
  - log10 on period, depth and prad; z-score fit on the training part only
  - 64-32 ReLU MLP, dropout 0.2, logit BCE, Adam 1e-3, batch 128
  - early stopping on a star-grouped inner validation split (patience 25)
- **Test split sealed.** All work uses the 8,288-KOI **pool** (train + val of the split-seed-42 split; 7,135 stars). The 913-KOI test split was never touched.
- **Cross-validation:** 5-fold, folds grouped by star (`kepid`) and stratified by system type (all-planet / all-FP / mixed).
  - Repeated over 5 fold shuffles × 10 model seeds = **250 models**.
  - Every pool KOI gets 50 out-of-fold (OOF) predictions, each from a model that never saw its star.
- **Notebook-style check:** 40 more runs on a single 80/10 split, to measure what one notebook run can and can't show.
- **Runtime:** about 2 s per model on the laptop CPU; the full sweep takes ~17 min.

## Step 1: noise floor

### Headline numbers

| Quantity | ROC-AUC |
|---|---|
| Pooled OOF, 50-model ensemble per KOI | **0.9251** (95% star-bootstrap CI 0.919–0.931) |
| Pooled OOF, CONFIRMED vs FP only | 0.964 |
| Mean single model on one fold (~1,660 KOIs) | 0.9227 ± 0.0047 |
| exo100926 (single run, val split 42) | 0.928, a typical draw |

### Where the spread comes from (1σ, ROC-AUC)

| Source | Std |
|---|---|
| Single model on one fold, all sources | 0.0047 |
| Fold composition (10-seed ensemble per fold) | 0.0046 |
| Model seed only, fold fixed | 0.0014 |
| Pooled OOF, one seed | 0.0009 |
| Pooled OOF, 10-seed ensemble | 0.0006 |
| **Two runs of the same config** (5 vs 5 seeds, same folds) | **0.00026** (95% of \|Δ\| < 0.0005) |

### Notebook-style single runs (~930-KOI validation set)

| Varying | Mean | Std | Range |
|---|---|---|---|
| Model seed (split fixed at 42), 20 runs | 0.9264 | 0.0016 | 0.923–0.930 |
| Which stars are in validation (model seed fixed), 20 re-splits of the pool | 0.9273 | **0.0072** | **0.914–0.943** |

### Takeaways

1. **One notebook run can't rank models.** Luck in which stars land in validation dominates. Two single runs would need to differ by about 0.02 before the difference means anything.
2. **Repeated grouped CV can.** On identical folds, two runs of the same config agree to within 0.0005. Training noise is no longer the limit; finite-sample uncertainty about new stars is, at about ±0.006 for this pool.
3. **Seed ensembling is nearly free but small:** +0.0017 AUC, saturating around 5 seeds (0.9224 → 0.9239 → 0.9241 at 10).
4. **The sealed test set has ±0.009 (1σ) sampling error** on its own, estimated from random test-sized star subsets of the pool. Use it once, as a final check, never to choose between models.
5. **Training stability:**
   - Early stopping lands at epoch 26–200 (median about 100). A few runs hit the 200 cap, which could be raised.
   - Per-KOI scores vary by ~0.03 between models, so per-KOI statements should use the ensemble score.

### Comparison protocol for steps 3–5

1. Run the candidate config with `cv_run(pool; repeats = 5, seeds = 1:5, ...)`, about 8 min. Keep the default `fold_seed0` so the folds match the baseline.
2. Run `compare_cv(new, baseline, pool; null = null_differences(baseline, pool))`.
3. Call it an improvement only if **both** checks pass:
   - |ΔAUC| is outside the training-noise null (`p_null` small);
   - the paired star-bootstrap 95% CI on ΔAUC excludes 0.

Sanity check: seeds 1–5 vs 6–10 of the baseline itself gives Δ = +0.0002, CI [−0.0002, +0.0006] and `p_null` = 0.47, so it correctly passes neither check.

## Step 2: error analysis (pooled OOF scores, threshold 0.5)

Overall: 666 FPs are called planets (16% of FPs) and 643 planets are called FPs (14%). Robovetter flags, comments and `koi_score` were used **for analysis only**.

### 1. Supplemental (non-DR25) KOIs are the largest single error source

| TCE source | KOIs | ROC-AUC | Error rate | Share of all errors | Median SNR |
|---|---|---|---|---|---|
| DR25 (`q1_q17_dr25_tce`) | 7,199 | **0.948** | 12% | 67% | 27.3 |
| Q1–Q16 only | 570 | **0.558** | 48% | 21% | 7.9 |
| DR24 only | 296 | 0.781 | 30% | 7% | 10.7 |
| No TCE | 223 | 0.740 | 33% | 6% | 10.2 |

- **What they are:** KOIs whose signal the final, uniform DR25 pipeline never re-detected. They have no `koi_score`, their parameters come from older fits, and they sit near the detection limit.
- **Why it matters:** they are 13% of the data but a third of the errors. Whether to keep, separate or drop them is a **scoping decision** for step 4.

### 2. Eclipsing binaries are largely solved

- Pure stellar-eclipse (SS-only) FPs are missed only 4.5% of the time.
- Odd/even depth, V-shape, `SWEET_EB` and `PLANET_IN_STAR` FPs are caught ≥ 97% of the time.
- Secondary-eclipse codes (`MOD_SEC_*`, `HAS_SEC_TCE`) are caught 91–93% of the time.

### 3. Centroid offsets, contamination and "not transit-like" FPs are the blind spot

| FP group | n | Missed |
|---|---|---|
| CO only (centroid offset) | 464 | **38%** |
| NT only (not transit-like) | 849 | 25% |
| EC only (ephemeris match / contamination) | 111 | 23% |
| No flag | 40 | 35% |
| SS only | 1,307 | 4.5% |

- CO and/or EC flags are on 58% of all missed FPs.
- The most-missed robovetter comment codes:
  - `CLEAR_APO` (43%) and `INDIV_TRANS_SKYE` (40%)
  - `MOD_NONUNIQ_ALT` (26%)
  - `CENT_RESOLVED_OFFSET` (22%) and `HALO_GHOST` (19%)
- All of these need pixel, centroid or per-transit information that the 11 catalog features don't contain. That is what Phase 2 is for.

### 4. Learned priors: mostly right, sometimes costly

- **SNR is the main variable.** Within-bin AUC goes from ~0.67 at SNR ≈ 7 to ~0.95 above SNR ≈ 30. SNR < 15 accounts for 62% of missed FPs and 58% of missed planets.
- **Multiplicity:**
  - FPs in systems with 3+ KOIs are missed 74–100% of the time (koi_count = 1: 14%).
  - 94% of missed planets are in single-KOI systems.
  - The model leans hard on `koi_count`.
- **Ultra-short periods (< 1.5 d):** most real planets there are called FPs. This costs real USP planets, e.g. many of the lowest-scored CONFIRMED planets are P < 1.5 d and R ≈ 1–2.7 R⊕.
- **Long periods (> 100 d):** within-bin AUC drops to ~0.63 (few transits).
- **Large radius / deep / grazing:** called FP almost always. That's correct for EBs, and it flags suspicious CANDIDATEs.

### 5. Most "missed planets" are CANDIDATEs, and many look like label noise

| Disposition | n | Median score | Score < 0.5 |
|---|---|---|---|
| CONFIRMED | 2,473 | 0.93 | 4.8% |
| CANDIDATE | 1,690 | 0.71 | 31% |
| FALSE POSITIVE | 4,125 | 0.09 | (84% correct) |

- 82% of missed planets are CANDIDATEs.
- The lowest-scored CANDIDATEs come in two kinds:
  - Mostly stellar-companion lookalikes: 25–700 R⊕, depths up to 35%, impact > 1, often with `koi_score = 0`. These are likely label noise.
  - Small ultra-short-period signals.

### 6. Comparison with the robovetter (`koi_score`, analysis only)

- **Coverage:** 7,199 pool KOIs have a `koi_score`.
- **Agreement:** Spearman 0.75 between the two scores; they make the same hard call on 86% of KOIs.
- **AUC on this subset:** robovetter 0.970 vs our model 0.948.
  - This gap is partly circular, since `koi_score` fed into many dispositions.
  - It's also roughly the value of the pixel-level and light-curve tests we don't have.

## Robustness checks

### Noise thresholds by metric

Use these when comparing configs later.

| Metric | Baseline | Same-config gap, 95% | Sampling 95% CI (star bootstrap) |
|---|---|---|---|
| ROC-AUC | 0.9251 | 0.0005 | 0.920–0.931 |
| PR-AUC | 0.9218 | 0.0006 | 0.914–0.929 |
| Log-loss (lower = better) | 0.3463 | 0.0011 | 0.334–0.358 |
| ROC-AUC, CONFIRMED vs FP | 0.9638 | 0.0005 | 0.959–0.968 |
| PR-AUC, CONFIRMED vs FP | 0.9381 | 0.0011 | 0.929–0.947 |

### Stability across the 5 independent fold shuffles

Ensemble rate shown with its range across the 5 fold shuffles and a binomial (Wilson) 95% interval.

| Group | n | Rate | Per-shuffle range | Wilson 95% |
|---|---|---|---|---|
| SS-only FPs missed | 1,307 | 4.5% | 4.4–4.6% | 3.5–5.8% |
| CO-only FPs missed | 464 | 37.9% | 37.3–39.9% | 33.6–42.4% |
| NT-only FPs missed | 849 | 24.6% | 24.9–27.6% | 21.8–27.6% |
| EC-only FPs missed | 111 | 23.4% | 21.6–23.4% | 16.5–32.1% |
| FPs with koi_count ≥ 3 missed | 43 | 79% | 77–79% | 65–89% |
| FPs with koi_count = 1 missed | 3,860 | 14.2% | 14.3–15.1% | 13.2–15.4% |
| CONFIRMED called FP | 2,473 | 4.8% | 4.6–5.2% | 4.0–5.7% |
| CANDIDATE called FP | 1,690 | 31.1% | 29.9–31.1% | 28.9–33.3% |
| Planets with P < 1.5 d called FP | 201 | **69.7%** | 66.7–68.2% | 63.0–75.6% |

AUC by detection source is also stable across shuffles: DR25 KOIs 0.946–0.947, Q1–Q16-only KOIs 0.54–0.56.

Single-shuffle rates are a little higher than the 50-model ensemble rate for the NT group, because averaging pulls borderline scores away from 0.5.

### DR25-detected KOIs only (7,199)

- **ROC-AUC 0.948** (95% CI 0.943–0.952); CONFIRMED vs FP **0.970**.
- **These findings persist:**
  - the CO-only blind spot (36% missed);
  - the multiplicity effect (FPs in 3-KOI systems missed 71%; missed planets are almost all singles);
  - easy EB-type FPs (4.5% missed);
  - the USP blind spot (76% of planets at the shortest periods called FP).
- **These shrink:**
  - NT-only misses (18% vs 25%);
  - the long-period weak spot (within-bin AUC ≈ 0.86 vs 0.63);
  - the low-SNR weak spot (≈ 0.80 vs 0.67 in the lowest bin).

  These were partly a supplemental-KOI effect.

## Recommendations (for steps 3–5, not done tonight)

1. **Step 3:** logistic regression and gradient-boosted trees (EvoTrees.jl) through the CV protocol above.
2. **Step 4:**
   - `koi_count` ablation;
   - log-transform `koi_model_snr`;
   - `koi_insol` back in;
   - decide on centroid observables (`koi_dicco_*`, `koi_dikco_*`, `koi_fwm_*`): they target the biggest blind spot, but they are robovetter inputs, which matters for an "independent" study;
   - decide how to treat supplemental KOIs (DR25-only training/evaluation, separate reporting, or a DR25-detected indicator).
3. **Labels:** try training on CONFIRMED vs FP only and scoring CANDIDATEs as an application.
4. **Step 5:** calibration and threshold choice on the pooled OOF scores (8,300 KOIs, not 930).
5. **Phase 2:** light curves and centroid time series aim directly at the CO / NT / EC failures and the low-SNR regime.

## Files

| Path | Contents |
|---|---|
| `src/VettingSetup.jl` | Core pipeline (unchanged behaviour; `split_by_star` now allows 2-way splits) |
| `src/CrossValidation.jl` | `star_folds`, `cv_run`, `noise_summary`, `null_differences`, `compare_cv`, `star_bootstrap`, `fast_auc` |
| `src/ErrorAnalysis.jl` | Flag / comment / binned / disposition / `koi_score` analyses and plots |
| `sweeps/cv_noise_floor.jl` | The headless sweep (250 CV models + 40 single-split runs) |
| `sweeps/results/cv_baseline_r5s10.jld2` | All OOF predictions, folds and per-model metrics: **the baseline for future comparisons** |
| `sweeps/results/single_split_runs.jld2` | Notebook-style runs |
| `sweeps/results/oof_baseline_r5s10.csv` | Per-KOI OOF score, score std, disposition, flags |
| `sweeps/nbcells/exo101026.jl` | Source of the notebook (percent format) |
| `sweeps/tools/run_nb.jl` | Headless notebook runner: `julia --project=. sweeps/tools/run_nb.jl <cells.jl> <out.ipynb>` |
| `figures/exo101026_*.png` | All figures |
