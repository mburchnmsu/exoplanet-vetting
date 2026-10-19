# Phase 2 plan: adding Kepler light curves to the vetting model

*Drafted 10/13/2026, after Phase 1 was finished and tested. **10/19/2026: the user approved all seven
recommendations (D1–D7) as written below; M0 started the same day.***

## 1. Goal and what counts as success

Phase 1 classifies KOIs from catalog numbers only. Phase 2 gives the model the **light curve itself**: the shape
of the transit, odd/even depth differences, secondary eclipses and (optionally) centroid motion during transit.
These are the things the catalog summarises away, and they are exactly where Phase 1 still fails.

**Targets to beat (Phase 1, pooled CV, DR25 KOIs; sealed-test values in brackets):**

| Model | DR25 ROC-AUC | 98%-recall point: FPs kept |
|---|---|---|
| Catalog-only (MLP + trees) | 0.955 [test 0.957] | 35% [test 31%] |
| Catalog + centroid | 0.983 [test 0.988] | 17% [test 16%] |
| Robovetter `koi_score` (reference) | 0.970 [test 0.979] | — |

**Remaining blind spots (catalog + centroid model unless stated, threshold 0.5):**

| Group | Phase 1 result | Why light curves should help |
|---|---|---|
| "Not transit-like" FPs (NT flag) | 27% missed | Shape: V-shapes, sinusoids, single-event glitches are visible in the folded curve |
| Contamination FPs (EC flag) | 25% missed | Partly ephemeris matching; partly odd/even and shape |
| Supplemental (non-DR25) KOIs | AUC 0.75 (catalog-only 0.67) | They have light curves too; the catalog columns are the weak part for them |
| Ultra-short-period planets | 32% called FP (catalog-only 58%) | A clean, flat-bottomed transit is direct evidence |
| Centroid-offset FPs (catalog-only model) | 36% missed | Centroid time series (only in the "+ centroid" family) |

**Success criteria (judged by CV with the existing paired protocol):**
1. **Main:** a light-curve model beats the matching Phase 1 model on DR25 KOIs, with a paired star-bootstrap CI
   that excludes 0. Done separately for the catalog-only family and the "+ centroid" family.
2. **Blind spots:** NT and EC miss rates fall clearly (for example, below 15%), and supplemental-KOI AUC rises.
3. **No regressions:** CONFIRMED planets lost at the 98% point do not rise.
4. **Stretch:** the catalog-only family (no DV centroid statistics) reaches the robovetter's 0.970 on DR25 KOIs.

For orientation only: published light-curve vetters (AstroNet, Shallue & Vanderburg 2018; ExoMiner, Valizadegan
et al. 2022) report AUCs around 0.99. But they classify TCEs (including many obvious non-transits), with different
labels, so their numbers are **not** comparable to ours.

## 2. Decisions (all recommendations approved 10/19/2026)

| # | **DECISION** | Recommendation | Why |
|---|---|---|---|
| D1 | Which machine runs Phase 2? | **Start the data download on this PC at low priority**; move training later if wanted | The download is network-bound and light on CPU/RAM, so it coexists with zodi. The 8 GB PC can take over once the compact cache exists (see §3.4) |
| D2 | What to keep on disk | **A compact per-star cache (~5–10 GB)**, not the raw FITS (~50–100 GB) and not only the final views | Re-download is slow (hours); views will be rebuilt many times while the pipeline is tuned |
| D3 | First model type | **Phase 2a: hand-made light-curve features + the existing trees/MLP first**, then **Phase 2b: a CNN** | This laptop has no CUDA GPU. 2a is cheap, interpretable, and plugs straight into the CV protocol; 2b is the bigger but slower bet |
| D4 | Centroid time series | **Yes, but only in the "+ centroid" family**, same rule as Phase 1 | Keeps the catalog-only family independent of centroid information |
| D5 | Final test for Phase 2 | **Reuse the same sealed test split, once**, for the final Phase 2 models | Phase 2 is never tuned on it, so the leak is minimal (we only saw Phase 1's aggregate scores). It keeps Phase 1 vs 2 on identical KOIs. A fresh split would cost training data and comparability. Disclose this in the write-up |
| D6 | CV budget for CNNs | **Screen with 1 fold shuffle × 2 seeds; final comparison with 5 shuffles × 3 seeds** (same folds as before) | 5 × 10 = 250 CNNs would take days on this CPU; 5 × 3 keeps the paired protocol at 1/3 the cost |
| D7 | Light-curve data in git | **No**: `data/lightcurves/` and view caches are git-ignored; code, small results and notebooks are tracked | Size (GBs), and they can be regenerated |

## 3. Data

### 3.1 Source
- **Kepler long-cadence light curves, final processing (DR25), quarters Q1–Q17**, from the MAST archive
  (`https://archive.stsci.edu/missions/kepler/lightcurves/<first 4 digits>/<9-digit KIC>/kplr<KIC>-<time>_llc.fits`).
  One FITS file per star per quarter (up to 18 per star).
- **Columns used:** `TIME` (BKJD, the same time system as `koi_time0bk`), `PDCSAP_FLUX` + error (systematics-corrected
  flux), `SAP_QUALITY` (cadence flags), `MOM_CENTR1/2` (flux-weighted centroids, for the "+ centroid" family only).
- **Stars:** all 7,927 stars of the cleaned catalog (pool + test). Test stars are downloaded and processed by the
  same code, but nothing about their labels is looked at until the one final evaluation.
- **Supplemental KOIs:** they come from the same light curves, so they are fully covered. This is one reason Phase 2
  may help them most.

### 3.2 Size and time (to be confirmed by the pilot, M0)
- Roughly 130k–140k FITS files, likely **50–100 GB** raw.
- At a polite 4 parallel requests, roughly **4–10 hours**, so it is an overnight job. It must be resumable (skip
  files that are already done) and back off on HTTP errors.
- Disk: 356 GB free on this PC, so raw files fit if needed. On the 8 GB PC, stream instead (download a star → reduce
  to the cache → delete the FITS).

### 3.3 Download design (`sweeps/p2_download.jl`)
- Per star: read the MAST directory listing → download each `_llc.fits` → extract the needed columns → write the
  star's compact record → delete (or keep, per D2) the raw files.
- A progress table (`data/lightcurves/manifest.csv`): star, quarters found, bytes, status, error text.
- It runs at below-normal priority with few threads, so zodi keeps the machine, and it can be killed and restarted
  at any point.

### 3.4 Compact cache
- One JLD2 per star, or a few shards: time, flux, flux error, quality, centroid row/col (Float32/Float64 as needed),
  and the quarter index. Roughly 1 MB per star, so about **5–10 GB** in total.
- Everything downstream (detrending, views, features) reads only the cache.

### 3.5 Data QA (pool stars only; no labels used)
- Coverage per KOI: number of quarters, number of transits actually observed, fraction of in-transit cadences
  flagged.
- KOIs with fewer than 2 observed transits, or no data at all, get a QA flag, and later a separate result row.
- Ephemeris sanity: does the folded transit sit at phase 0 with roughly the catalog depth? A mismatch flags a bad
  period or epoch.

## 4. Preprocessing

### 4.1 Cleaning and detrending (`src/Detrend.jl`)
1. Drop cadences with non-zero quality flags (standard bitmask) and non-finite flux; normalise each quarter by its
   median.
2. **Mask the in-transit points of every KOI on the star** (using the catalog ephemerides and 1.5× duration) before
   fitting the trend.
3. **Detrend with a robust sliding filter**: a Tukey biweight in a window of max(0.5 d, 3 × duration), the default
   of the `wotan` package (Hippke et al. 2019). It is implemented directly in Julia, so no new dependency.
4. Divide by the trend; sigma-clip upward outliers only (never clip downward, which would eat transits).

### 4.2 Views (`src/Views.jl`)
All views are built from the catalog period, epoch and duration (allowed inputs, as in Phase 1). When building one
KOI's views, the other KOIs on the same star are masked out.

| View | Bins | Span | Shows | Family |
|---|---|---|---|---|
| **Global** | 2001 | whole orbit | Secondary eclipses, phase variations, other dips | both |
| **Local** | 201 | ±2 transit durations | Transit shape (U vs V), ingress/egress | both |
| **Secondary** | 201 | ±2 durations around the deepest off-transit dip | EB secondaries | both |
| **Odd / even** | 201 each | like local, odd and even transits separately | EBs at twice the true period | both |
| **Centroid** | 201 | like local, centroid shift in transit | Background EBs (CO) | + centroid only |

- **Binning:** median per bin; empty bins are filled by interpolation, and the fraction of empty bins is kept as a QA
  feature.
- **Normalisation:** the local view is set to 0 out of transit and −1 at the transit minimum (as in AstroNet). Odd/even
  and secondary views use the **same scale** as the local view, so relative depths survive. The absolute depth stays
  available through the catalog features.

### 4.3 Hand-made light-curve features for Phase 2a (`src/LCFeatures.jl`)
These are computed by our own pipeline, so they are independent of the robovetter's numbers.
- **Odd/even depth difference** and its significance (σ).
- **Secondary eclipse** depth, its significance, and its phase.
- **Transit shape:** a trapezoid fit to the local view gives the ingress/egress-to-total duration ratio (V-shape
  indicator) and the fitted depth vs the catalog depth.
- **Single-event dominance:** the deepest single transit vs the mean (catches one-off glitches).
- **Noise and coverage:** out-of-transit scatter, number of observed transits, fraction of empty bins.
- **"+ centroid" family only:** in-transit centroid shift and its significance, computed from the light-curve
  files' `MOM_CENTR1/2` columns.
- **Only a few (~10) features**, each with a stated reason, so the tree/MLP CV protocol can run unchanged (it is fast).

## 5. Models (in order)

| Step | Model | Inputs | Compared with |
|---|---|---|---|
| **2a-1** | Trees + quantile MLP (Phase 1 recipes) | catalog + LC features | Phase 1 catalog-only (0.955) |
| **2a-2** | Same | catalog + LC + DV centroid + our centroid features | Phase 1 + centroid (0.983) |
| **2b-1** | Small 1D CNN, light curve only | global + local views | 2a-1 (to see what the raw views add) |
| **2b-2** | Hybrid CNN | views + catalog features joined before the dense layers | 2a-1 |
| **2b-3** | Hybrid CNN + centroid | + secondary, odd/even, centroid views; + centroid features | 2a-2 |
| **2c** | Ensemble | average of the best 2a and 2b models (as in Phase 1, where averaging gave a real gain) | best single model |

**CNN design (2b), sized for a 4-core CPU:**
- Two convolution towers in the AstroNet style. The global tower has 3–4 blocks of [Conv1D(k = 5) ×2 → max-pool],
  with 8 → 64 channels. The local tower has 2 blocks with 16 → 32 channels.
- The outputs are flattened and joined with the quantile-transformed catalog features → Dense(64) → dropout → 1
  output.
- Target size **under ~200k parameters**: there are only ~7k training KOIs per fold, and the CPU is the limit.
- **Training:** AdamW, early stopping on the same inner split as Phase 1, and mirror augmentation (time-reversed
  views).
- **Timing gate:** before any CV, time one epoch. If one model takes more than ~10 min, shrink the global view
  (2001 → 1001 bins) or the towers before going on.

## 6. Evaluation protocol (unchanged from Phase 1 wherever possible)
- **The same pool, folds (`fold_seed0 = 1000`) and star grouping**, so every Phase 2 run is paired with the Phase 1
  out-of-fold predictions.
- **Comparisons:** `compare_cv` (paired star bootstrap, plus the training-noise null). Headline on DR25 KOIs;
  supplemental KOIs always reported separately; provenance never an input.
- **Breakdowns:** by robovetter flag (NT/SS/CO/EC), period (USP), SNR, number of transits observed, supplemental vs
  DR25, and the QA-flagged KOIs.
- **Calibration and operating points:** reuse `src/Calibration.jl` exactly as in step 5 (nested, threshold menu,
  98% planet recall as the headline).
- **Final:** freeze → one scoring of the sealed test split (per D5) → report next to the Phase 1 test numbers.

## 7. Compute and resources

| Stage | CPU | RAM | Time (estimate) |
|---|---|---|---|
| Download + compact cache | low (network-bound) | < 1 GB | 4–10 h, overnight |
| Detrending + views, all KOIs | moderate, parallel per star | ~1–2 GB | under 1 h |
| Phase 2a CV (trees + MLP, 5 × 10) | as in Phase 1 | ~2 GB | ~10–20 min per model family |
| Phase 2b CNN, per model | heavy | ~1–3 GB | to be measured (gate in §5) |
| Phase 2b final CV (5 × 3 = 75 CNNs) | heavy | ~2 GB | likely a full night |

- **Coexisting with zodi:** everything runs at below-normal priority with 2–4 threads, as Phase 1 did tonight.
  Background jobs can be killed under memory pressure, so every script stays resumable (per fold or per star).
- **The 8 GB PC** can run everything from the compact cache onwards. The CNN CV would be slower there; measure first.

## 8. Milestones and go/no-go gates

| Milestone | Work | Gate to continue |
|---|---|---|
| **M0 Pilot** (1 session) | Download ~50 stars (planets, EBs, CO FPs, NT FPs; pool only), build the cache, detrend, plot views | Pipeline works end to end; size and time estimates confirmed; views look right by eye |
| **M1 Download** (overnight) | All 7,927 stars, resumable | ≥ 98% of KOIs have data; failures listed |
| **M2 Views + QA** | Detrend + all views; QA notebook: coverage, ephemeris check, **average view per class** (pool only) | EB average shows a secondary; planet average is U-shaped; no systematic artifacts |
| **M3 Phase 2a** | LC features; CV for both families | Paired Δ vs Phase 1 reported; blind-spot table |
| **M4 Phase 2b** | Timing gate; CNN screens (1 × 2); best one at 5 × 3 | Beats 2a, or adds to it in an ensemble (CI excludes 0) |
| **M5 Final** | Ensemble, calibration, operating-point menu, freeze | All of §1's criteria checked |
| **M6 Test** | One-time test scoring; write-up next to Phase 1 | — |

Notebooks keep the usual naming (`exoMMDDYY.ipynb` in `notebooks/`), one per working day, with M0–M2 likely in
the first.

## 9. Risks and how to handle them

| Risk | Mitigation |
|---|---|
| MAST slow, rate-limited or down | Polite concurrency, retries with back-off, resumable manifest; run overnight |
| Machine memory pressure (zodi) kills jobs | Streaming per star, small memory footprint, resumable everything, below-normal priority |
| CNN too slow on CPU | Phase 2a first (likely most of the gain); small towers; fewer seeds (D6); shorter global view |
| Wrong catalog ephemerides smear the views | QA ephemeris check (M2); flag and report those KOIs separately |
| Few observed transits (long periods) → noisy views | Record the number of transits as a feature; report that group separately |
| **Circularity:** the robovetter's flags were made from these same light curves | Expected and fine: we learn from the data, not from its flags. Never use flags or `koi_score` as inputs. Say so in the write-up |
| Overfitting (only ~7k training KOIs per fold) | Small models, augmentation, early stopping, weight decay; the paired CV protocol catches it |
| Detrending distorts long transits | The window is ≥ 3 × duration with in-transit masking; check average views of long-duration planets in M2 |

## 10. Code layout (new files)

| Path | Contents |
|---|---|
| `src/LightCurves.jl` | MAST listing/download, FITS reading, compact cache read/write |
| `src/Detrend.jl` | Quality masking, KOI in-transit masks, biweight detrending |
| `src/Views.jl` | Folding, binning, global/local/secondary/odd-even/centroid views, normalisation |
| `src/LCFeatures.jl` | Phase 2a hand-made features |
| `src/CNN.jl` | Tower architecture, `train_cnn` (plugs into `cv_run` via `trainer =`, like `train_evotree`) |
| `sweeps/p2_download.jl` | Resumable download (+ manifest) |
| `sweeps/p2_build.jl` | Cache → detrend → views + features → `data/views_*.jld2` |
| `sweeps/p2_cv_*.jl` | CV runs for 2a / 2b, resumable |
| `data/lightcurves/` | Cache + manifest (**git-ignored**) |

- **New package:** FITSIO.jl (it brings its own CFITSIO binary, so no system install is needed), added with
  `PRESERVE_ALL` as before.
- `cv_run` needs one small generalisation: it must pass view arrays as well as the feature matrix to the trainer.
  The fold logic stays untouched.

## 11. First session checklist (M0)
1. Get the user's calls on D1–D7, or accept the recommendations.
2. Add FITSIO.jl; write the minimal `LightCurves.jl` + `Detrend.jl` + `Views.jl`.
3. Pick ~50 pool stars covering each class (CONFIRMED, CANDIDATE, EB/SS, CO, NT, a USP planet, a long-period
   planet, a supplemental KOI).
4. Download → cache → views; time and size per star → extrapolate M1.
5. Notebook: raw vs detrended light curves and the views for 6–8 example KOIs.
6. Commit; schedule M1 overnight (at low priority) if the gate passes.
