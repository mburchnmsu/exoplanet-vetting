# Kepler vetting: noise floor, error analysis and baselines

*Steps 1–2: 10/10/2026 · notebook `exo101026.ipynb` · compute `sweeps/cv_noise_floor.jl`*
*Step 3: 10/13/2026 · notebook `exo101326.ipynb` · compute `sweeps/cv_baselines.jl` · results in `sweeps/results/`*

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

## Step 3: baselines (10/13/2026, notebook `exo101326.ipynb`)

Same folds and protocol as the MLP baseline (5 fold shuffles × 10 seeds, `compare_cv`). Compute: `sweeps/cv_baselines.jl`.

| Model | Pooled OOF ROC-AUC | Δ vs MLP [95% CI] | Log-loss | CONFIRMED vs FP | Folds beating MLP |
|---|---|---|---|---|---|
| MLP (baseline) | 0.9251 | — | 0.346 | 0.964 | — |
| Logistic regression | 0.8668 | −0.058 [−0.064, −0.053] | 0.456 | 0.894 | 0 / 25 |
| Trees (EvoTrees), untuned | 0.9306 | +0.0056 [+0.0032, +0.0080] | 0.336 | 0.973 | 23 / 25 |
| **Trees, tuned** (depth 8, min_weight 10, L2 10, colsample 0.8) | **0.9320** | **+0.0069 [+0.0045, +0.0094]** | **0.331** | **0.973** | **25 / 25** |
| MLP + tuned trees, 50/50 average | 0.9321 | +0.0070 (vs trees: +0.0001, CI includes 0) | — | — | — |

- **Trees beat the MLP for real:** more than 10× the training-noise bar, on every fold, and on log-loss too.
  - Untuned trees already win, so this does not depend on tuning.
  - Tuning was scored on separate fold shuffles (`fold_seed0 = 5000`).
- **The nonlinearity matters:** logistic regression is 0.058 behind.
- **Tuning:** only tree depth matters (0.927 at depth 3 → 0.932 at depth 8). Depth 8 is the edge of the grid.
- **No blind spot is fixed** (threshold 0.5):

  | Group | MLP | Tuned trees |
  |---|---|---|
  | CO-only FPs missed | 37.9% | 37.5% |
  | NT-only FPs missed | 24.6% | 26.0% |
  | EC-only FPs missed | 23.4% | 22.5% |
  | FPs with koi_count ≥ 3 missed | 79% | 79% |
  | Q1–Q16-only KOIs, AUC | 0.56 | 0.58 |

- **Where trees gain:**
  - DR25 KOIs: AUC 0.953 vs 0.948
  - CONFIRMED called FP: 3.6% vs 4.8%
  - Planets with P < 1.5 d called FP: 59% vs 70%
- **The two models mostly agree:** Spearman 0.967; they make a different hard call on 6.5% of KOIs. Averaging them adds nothing over trees alone.
- **Calibration:** both are well calibrated (reliability plot in the notebook).
- **What the models use:**
  - Tree importance: prad > period > duration > koi_count > SNR.
  - Biggest logistic-regression coefficients: teq (−), prad (−), koi_count (+).
- **Conclusion:** the remaining errors are limited by the information in the 11 catalog features, not by the model class. Step 4 features and Phase 2 light curves should matter more than architecture.

### Follow-up: closing the MLP gap by tuning (`sweeps/cv_mlp_gap.jl`, notebook section 7)

Decision (10/13): carry **both** models forward, and first try to close the MLP's gap by tuning.

- **Screen:** 11 single changes and 3 combinations, on the tree-tuning folds. The baseline MLP scores 0.9255 there; the best tree 0.9318.
  - **Small real gains:** learning rate 3e-3 (+0.0009) and a 600-epoch cap with patience 50 (+0.0008).
  - **No effect** (|Δ| ≤ 0.0007, CIs include 0): wider layers (128-64, 256-128), a third layer, dropout 0.1 / 0.3, batch 64.
  - **Weight decay** (AdamW, wd 0.01 / 0.1) slightly hurts.
- **Best: `lr3_e600`** (learning rate 3e-3, epoch cap 600, patience 50). On the standard folds (5 × 10):

  | Comparison | Δ ROC-AUC [95% CI] |
  |---|---|
  | vs baseline MLP | **+0.0014** [+0.0009, +0.0019], p_null < 0.002 |
  | vs tuned trees | −0.0055 [−0.0080, −0.0032] |

  - Its ROC-AUC is **0.9265** and log-loss 0.343 (vs 0.346 for the baseline).
  - It **closes ~20% of the gap**.
  - No run reaches the new cap (median best epoch 59).
- **Interpretation:** the gap isn't about training length or capacity. Likely cause: the **input representation**.
  - Trees only see the order of each feature's values, so skewed inputs don't affect them.
  - The MLP sees z-scored values, with only period, depth and prad logged; SNR, duration, teq, srad and koi_count are strongly skewed.
  - Testing log or quantile transforms belongs to step 4.
- Saved: `sweeps/results/mlp_screen/` and `mlp_screen.csv` (screen), `cv_mlp_lr3_e600_r5s10.jld2` (final).

Decision (10/13): adopt `lr3_e600` as the MLP reference and test the input-representation explanation (step 4, part 1).

## Step 4, part 1: input transforms (10/13/2026, notebook `exo101326.ipynb` section 8)

**The problem:** as the MLP saw them, several features were extremely skewed.

| Feature | Skew |
|---|---|
| impact | 23 (values up to 100 vs typical 0–1.3) |
| srad | 22 |
| duration | 6 |
| SNR | 5 |
| teq | 4 |

Some z-scored inputs reached 38σ. Two fixes were added to `prepare_data(...; transform)` and `cv_run(...; transform)`, both fit on the training part only:

- **`:log_more`:** also log10 duration, SNR (+1), teq, srad, koi_count, impact (+0.1). Max |z| 13.
- **`:quantile`** (rank-gauss): empirical CDF on the training part → standard-normal quantile. Max |z| 4.5.

Standard folds, 5 shuffles × 10 seeds (`sweeps/cv_mlp_gap.jl final ...`):

| Model | ROC-AUC | Log-loss | CONF vs FP | Δ vs tuned MLP | Δ vs tuned trees |
|---|---|---|---|---|---|
| MLP tuned (`lr3_e600`) | 0.9265 | 0.343 | 0.965 | — | −0.0055 [−0.0080, −0.0032] |
| MLP tuned + `log_more` | 0.9326 | 0.329 | 0.974 | +0.0062 [+0.0042, +0.0084] | +0.0006 [−0.0015, +0.0028] |
| **MLP tuned + `quantile`** | **0.9330** | **0.328** | **0.974** | **+0.0065** [+0.0042, +0.0092] | +0.0010 [−0.0012, +0.0031] |
| Trees, tuned | 0.9320 | 0.331 | 0.973 | +0.0055 | — |
| **Quantile MLP + trees, 50/50 average** | **0.9352** | **0.324** | — | — | **+0.0032** [+0.0021, +0.0043] |

- **The input representation was the whole MLP gap.**
  - The quantile MLP is +0.008 over the original MLP and tied with the trees.
  - Architecture tuning never mattered.
- **The two models now combine:** the average is +0.0022 [+0.0011, +0.0032] over the quantile MLP, and the best result so far. With the original MLP, averaging added nothing.
- **Blind spots shrink but remain** (threshold 0.5, original MLP → quantile MLP):

  | Group | Original MLP | Quantile MLP |
  |---|---|---|
  | Missed FPs overall | 16.1% | 14.3% |
  | NT-only FPs missed | 24.6% | 20.4% |
  | CO-only FPs missed | 37.9% | 35.1% |
  | FPs with 3+ KOIs missed | 79% | 74% |
  | Q1–Q16-only KOIs, AUC | 0.56 | 0.58 |
  | DR25 KOIs, AUC | 0.948 | 0.953 |

**New references for the rest of step 4:**
- MLP = `lr3_e600` + `transform = :quantile`;
- trees = tuned EvoTrees;
- combined model = their 50/50 average.

## Step 4, part 2: `koi_count`, `koi_insol`, deeper trees (10/13/2026, notebook section 9, `sweeps/step4_features.jl`)

**`koi_insol`: not run, on purpose.** Spearman(`koi_insol`, `koi_teq`) = 1.0000 on the pool, because teq is derived from insolation. Trees and the quantile MLP only see value order, so it adds exactly zero information. It stays out.

**`koi_count` ablation** (standard folds, 5 × 10; Δ = without − with):

| Model | With `koi_count` | Without | Δ ROC-AUC [95% CI] | Δ log-loss |
|---|---|---|---|---|
| MLP + quantile | 0.9330 | 0.9294 | −0.0036 [−0.0053, −0.0019] | +0.009 |
| Trees, tuned | 0.9320 | 0.9280 | −0.0040 [−0.0057, −0.0022] | +0.010 |
| MLP + trees average | 0.9352 | 0.9315 | −0.0037 [−0.0053, −0.0021] | +0.009 |

- **It is a pure multiplicity prior.**
  - Within each group (1 / 2 / 3+ KOIs), AUC is unchanged without it: 0.912 / 0.921 / 0.908 with, against 0.912 / 0.923 / 0.907 without.
  - Its value comes entirely from shifting scores between groups: planet fraction is 39% for singles, 81% for 2-KOI systems and 95% for 3+.
  - This matches the known multiplicity boost. It's legitimate, not leakage.
- **Trade-off** (average model, threshold 0.5, with → without):
  - FPs missed in 3+ systems: 77% → 51%
  - Planets called FP in 2- and 3+ systems: 2.9% / 1.2% → 6.2% / 3.8%
- **Decision: keep it.** FPs on multi-KOI stars stay the catalog model's hardest case (a Phase-2 target).

**Deeper trees:** depth 10 (+0.0001) and 12 (+0.0003) over depth 8 on the tuning folds, below the 0.0005 bar. Depth 8 stays.

## Step 4, part 3: centroid features and supplemental-KOI scoping (10/13/2026, notebook section 10, `sweeps/step4_scoping.jl`)

**Decisions (user, 10/13):**
- Centroid features are allowed, as a **separate "catalog + centroid" model** next to catalog-only.
- Supplemental KOIs stay in training. **Headline metrics are on DR25-detected KOIs**; supplemental KOIs are reported separately.
- Provenance is never a model input.

**Centroid features** (`src/Centroids.jl`, 7 features):
- From difference images: offsets from the out-of-transit centroid and from the KIC position, in arcsec and in σ.
- From flux-weighted centroids: motion significance, source offset, and in-transit shift.
- Missing values (5–10%) get the training median, with **no missing indicators**: missingness is informative (17% vs 52% planets) and total for no-TCE KOIs.

| Average model (MLP + trees) | Catalog-only | + centroid | Δ [95% CI] |
|---|---|---|---|
| DR25 KOIs (headline) | 0.955 | **0.983** | +0.028 [+0.024, +0.032] |
| All KOIs | 0.935 | 0.970 | +0.035 [+0.031, +0.039] |
| Supplemental KOIs | 0.668 | 0.755 | +0.087 [+0.064, +0.111] |
| KOIs with centroid data | 0.935 | 0.969 | +0.034 (so the gain is not a missing-data artifact) |

The MLP and the trees separately show the same pattern (+0.028 and +0.029 on DR25).

**Error rates, average model at threshold 0.5:**

| Group | Catalog-only | + centroid |
|---|---|---|
| CO-only FPs missed | 36% | **4.3%** |
| CO/EC FPs missed | 19% | 5% |
| All FPs missed | 15% | 10% |
| CONFIRMED called FP | 3.6% | 1.4% |
| Planets with P < 1.5 d called FP | 58% | 32% |
| NT-only FPs missed | 22% | 27% |
| EC-only FPs missed | 23% | 25% |

NT and EC are not helped.

**⚠ Circularity:**
- The robovetter set the CO flag by thresholding these same statistics, so part of the gain re-learns its rule.
- The centroid model even out-ranks `koi_score` on DR25 (0.983 vs 0.970).
- Keep the **catalog-only model as the independent reference** and always report the two side by side.

**DR25-only training** (Δ = DR25-only − train-all):

| Scored on | Δ ROC-AUC |
|---|---|
| DR25 KOIs | −0.0002 to +0.0007, all CIs include 0 |
| Supplemental KOIs | −0.03 to −0.04, all CIs exclude 0 |

→ **Keep training on all KOIs.**

**Headline table, DR25-detected KOIs (7,199), pooled OOF ROC-AUC:**

| Model | DR25 ROC-AUC [95% CI] | DR25 CONFIRMED vs FP | Supplemental |
|---|---|---|---|
| MLP baseline (exo100926) | 0.948 [0.943, 0.952] | 0.970 | 0.652 |
| MLP + quantile | 0.953 | 0.977 | 0.666 |
| Trees, tuned | 0.953 | 0.976 | 0.659 |
| **Catalog-only: MLP + trees average** | **0.955** [0.951, 0.960] | 0.978 | 0.668 |
| **Catalog + centroid: average** | **0.983** [0.981, 0.985] | 0.995 | 0.755 |
| Robovetter `koi_score` (reference, partly circular) | 0.970 | 0.984 | — |

**Step 4 status: done.**
- Next is step 5: calibration and threshold choice, for both the catalog-only and catalog + centroid models.
- Then a single final evaluation on the sealed test split.

## Step 5: calibration and operating points (10/13/2026, notebook section 11, `src/Calibration.jl`, `sweeps/step5_freeze.jl`)

**Setup.**
- Both final models: catalog-only and catalog + centroid (each = quantile MLP + tuned trees, 50/50 average).
- Calibration uses each fold shuffle's 10-seed mean (the size of ensemble we ship), not the 50-model repeat-mean.
- Everything is nested: calibrators and thresholds are fitted on the other folds and applied to the held-out fold.
- Thresholds are chosen on DR25 KOIs, one threshold for all KOIs (provenance never sets a KOI's threshold).
- User decisions: report a menu of operating points; open the sealed test set later, together.

**Calibration: not needed.**

| Model, DR25 KOIs | log-loss none / Platt / isotonic | ECE none / Platt / isotonic |
|---|---|---|
| Catalog-only | 0.275 / 0.275 / 0.277 | 0.018 / 0.014 / 0.012 |
| Catalog + centroid | 0.170 / 0.170 / 0.172 | 0.014 / 0.010 / 0.008 |

- A Platt fit on all OOF scores is almost the identity (a ≈ 1.04–1.05, b ≈ −0.02).
- Calibrators shave ECE slightly but don't improve log-loss or Brier; isotonic is slightly worse and costs AUC via ties.
- → **Frozen choice: no calibrator.** Scores read as planet probabilities on DR25-like KOIs.
- **Supplemental KOIs are miscalibrated (ECE 0.05–0.08).** The cause is weak ranking, not a base-rate shift, and a fix
  would need provenance as an input (ruled out). Their probabilities should be read as unreliable.

**Operating-point menu (nested estimates on DR25 KOIs, mean of 5 fold shuffles; per 1,000 DR25 KOIs, 51% planets):**

| Rule | Model | Frozen t | Recall | Precision | FPs kept | Planets lost /1000 | FPs kept /1000 | CONFIRMED lost |
|---|---|---|---|---|---|---|---|---|
| Recall 95% | catalog-only | 0.31 | 0.950 | 0.814 | 23% | 25 | 111 | 1.5% |
| | + centroid | 0.44 | 0.950 | 0.913 | 10% | 26 | 47 | 1.1% |
| Recall 98% | catalog-only | 0.15 | 0.980 | 0.745 | 35% | 10 | 172 | 0.4% |
| | + centroid | 0.21 | 0.980 | 0.856 | 17% | 10 | 84 | 0.4% |
| Recall 99% | catalog-only | 0.08 | 0.990 | 0.696 | 46% | 5 | 222 | 0.2% |
| | + centroid | 0.09 | 0.990 | 0.807 | 25% | 5 | 122 | 0.2% |
| Precision 95% | catalog-only | 0.78 | 0.741 | 0.949 | 4.2% | 133 | 20 | 13% |
| | + centroid | 0.61 | 0.916 | 0.950 | 5.1% | 43 | 25 | 2.1% |
| Precision 99% | catalog-only | 0.98 | 0.289 | 0.987 | 0.4% | 365 | 2 | 59% |
| | + centroid | 0.92 | 0.730 | 0.989 | 0.9% | 138 | 4 | 12% |
| Max F1 | catalog-only | 0.49 | 0.904 | 0.874 | 14% | 49 | 67 | 3.6% |
| | + centroid | 0.55 | 0.930 | 0.937 | 6.6% | 36 | 32 | 1.8% |
| p = 0.5 | catalog-only | 0.50 | 0.900 | 0.881 | 13% | 52 | 63 | 3.8% |
| | + centroid | 0.50 | 0.941 | 0.927 | 7.9% | 30 | 38 | 1.4% |

- Nested targets are hit exactly; threshold spread across fold shuffles ≤ 0.04.
- Supplemental KOIs at the same thresholds: see `sweeps/results/step5_operating_points.csv` (e.g. max F1:
  recall 0.53 / precision 0.56 catalog-only).
- **Frozen:** `models/step5_frozen.jld2` holds both final models (10 MLP + 10 tree seeds each, trained on the whole
  pool with a fixed inner early-stopping split), their preprocessing, and the menu thresholds. The test split was not
  touched.

**Headline operating point (user decision, 10/13):** **98% planet recall**, because the model is a triage filter
(a rejected planet is lost; a false positive only costs follow-up time). Best F1 is reported alongside for comparison
with the literature; ROC-AUC remains the main threshold-free number; the full menu is always shown.

**Step 5 status: done.**

## Sealed-test evaluation (10/13/2026, used ONCE; `sweeps/step5_test.jl`, notebook section 12)

The frozen models scored the test split (`split_by_star` seed 42: 913 KOIs, 792 stars, 796 DR25) once. Nothing was
fitted or tuned on it, and the script refuses to run again. Star-bootstrap 95% CIs.

| Model, DR25 KOIs | Test ROC-AUC | CV expected | Test CONFIRMED vs FP |
|---|---|---|---|
| Catalog-only | **0.957** [0.944, 0.970] | 0.954 | 0.975 |
| Catalog + centroid | **0.988** [0.982, 0.994] | 0.983 | 0.995 |
| Robovetter `koi_score` (reference) | 0.979 | — | — |

| Operating point (DR25 KOIs) | Model | t | Test recall | Test precision | FPs kept (test / CV) |
|---|---|---|---|---|---|
| **98% planet recall (headline)** | catalog-only | 0.145 | 0.988 [0.975, 0.997] | 0.771 [0.731, 0.809] | 31% / 35% |
| | + centroid | 0.211 | 0.993 [0.983, 1.000] | 0.869 [0.837, 0.898] | 16% / 17% |
| Best F1 | catalog-only | 0.491 | 0.899 [0.868, 0.929] | 0.892 [0.861, 0.923] | 11% / 14% |
| | + centroid | 0.548 | 0.963 [0.943, 0.980] | 0.947 [0.920, 0.968] | 5.6% / 6.6% |

- **CV is confirmed:** every CV estimate lies inside the test CI, for both models and every subset. The test split is
  slightly easier than the pool, within its noise.
- At the 98% threshold, no DR25 CONFIRMED planet in the test set is lost by either model.
- Supplemental KOIs (117): 0.72 / 0.79 (CV 0.66 / 0.75; wide CIs).
- Full tables: `sweeps/results/test_auc.csv`, `test_operating_points.csv`; predictions: `test_predictions.csv`.
- **The test set is now spent.** Later changes must be judged by CV and reported as post-test.

## Post-test: CONFIRMED-only training (10/13/2026, notebook section 13, `sweeps/labels_confirmed.jl`)

Same final recipes and folds, but CANDIDATEs are dropped from training (6,598 of 8,288 KOIs); every KOI still gets an
OOF score. CV only.

| Paired Δ (CONFIRMED-only − train-all) | Catalog-only | Catalog + centroid |
|---|---|---|
| DR25 CONFIRMED vs FP | 0.978 → 0.982, **+0.0041** [+0.0027, +0.0055] | 0.9945 → 0.9958, **+0.0013** [+0.0005, +0.0020] |
| DR25, CANDIDATE = planet (the task) | 0.955 → 0.941, **−0.0147** | 0.983 → 0.976, **−0.0071** |
| Supplemental CONFIRMED vs FP | −0.018 [−0.048, +0.004] | +0.003 [−0.024, +0.030] |

- DR25 CANDIDATEs below the 98%-of-CONFIRMED cut: 36% / 41% for the CONFIRMED-only models (train-all: 15% / 19%).
- The CONFIRMED-only ranking of CANDIDATEs agrees better with `koi_score` (Spearman 0.24 → 0.32; 0.33 → 0.38).
- Caveat: CONFIRMED planets are the *confirmable* ones (high SNR, multis), so a low score means "unlike confirmed
  planets", not necessarily "FP".
- **Not adopted.** The frozen train-all models stay the headline. CONFIRMED-only is a candidate secondary score for
  follow-up priority.

## Recommendations (for steps 4–5)

1. **Step 3:** done (see above).
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
| `sweeps/tools/run_nb.jl` | Headless notebook runner: `julia --project=. sweeps/tools/run_nb.jl sweeps/nbcells/<name>.jl notebooks/<name>.ipynb` (all dated notebooks live in `notebooks/`) |
| `figures/exo101026_*.png` | All figures |
| `src/Baselines.jl` | Step 3: `train_evotree` (EvoTrees, plugs into `cv_run(...; trainer = train_evotree)`), `tree_importance`, `logreg_coefs`; logistic regression = `train_model(...; hidden = ())` |
| `sweeps/cv_baselines.jl` | Step 3 sweep (resumable): logistic regression, untuned and tuned trees, 60-setting tree grid |
| `sweeps/results/cv_logreg_r5s10.jld2`, `cv_evotree_default_r5s10.jld2`, `cv_evotree_tuned_r5s10.jld2` | Step 3 CV runs (same folds as the baseline) |
| `sweeps/results/evotree_grid.csv` / `.jld2` | Tree tuning grid results |
| `sweeps/results/oof_baselines_r5s10.csv` | Per-KOI OOF scores for MLP, logistic regression and both tree models |
| `figures/exo101326_*.png` | Step 3 and 4 figures |
| `sweeps/cv_mlp_gap.jl` | MLP tuning screen and transforms (`screen` / `final` modes, resumable) |
| `sweeps/step4_features.jl` | `koi_count` ablation and deeper-tree screen |
| `sweeps/results/cv_mlp_lr3_e600*_r5s10.jld2` | Tuned MLP: plain, `log_more`, `quantile` (**quantile = MLP reference**) |
| `sweeps/results/cv_mlpq_nocount_r5s10.jld2`, `cv_evotree_nocount_r5s10.jld2` | `koi_count` ablation runs |
| `sweeps/results/mlp_screen/`, `mlp_screen.csv`, `evotree_depth_screen.csv` | Screening runs on the tuning folds |
| `src/Centroids.jl` | `CENTROID_FEATURES`, `add_centroid_features`, `is_dr25` (row selection / reporting only) |
| `sweeps/step4_scoping.jl` | Centroid models and DR25-only training runs |
| `sweeps/results/cv_mlpq_centroid_r5s10.jld2`, `cv_evotree_centroid_r5s10.jld2` | Catalog + centroid models |
| `sweeps/results/cv_mlpq_dr25train_r5s10.jld2`, `cv_evotree_dr25train_r5s10.jld2` | Trained on DR25 KOIs only |
| `src/Calibration.jl` | Step 5: `repeat_scores`, Platt / isotonic calibrators, `nested_calibrate`, `pick_threshold`, `nested_calls`, `ece`, `brier` |
| `sweeps/step5_freeze.jl` | Trains and saves the two final models + menu thresholds (`models/step5_frozen.jld2`) |
| `sweeps/results/step5_operating_points.csv` | The operating-point menu, incl. supplemental-KOI columns |
