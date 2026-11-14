# Phase 4 plan: the "not transit-like" (NT) blind spot

*Drafted 11/09/2026 after Phase 3 (`PHASE3_PLAN.md`). **Approved 11/13/2026** (user, via the hub): N1–N5 take the
recommended option in §3. The same day the user adopted the Phase 3 "+ ephemeris" models (frozen in
`models/p3_frozen.jld2`, post-test, CV-validated), so N4's second comparison (on the "+ ephemeris" lineage) is run.*

## 1. Where NT stands, and what the misses look like

NT false positives are signals that are not a transit at all: detrending residuals, stellar variability, rolling-band
noise, cosmic-ray or momentum-dump artifacts. They are the largest remaining error group:

| NT-only FPs (n = 849; DR25 433) | Catalog family | + centroid family |
|---|---|---|
| Missed at p = 0.5, Phase 2a / + ephemeris | 20.8 / 21.6% | 23.7 / 24.4% |
| Missed at 98% recall | 70.7 / 70.0% | 60.2 / 59.7% |

At the headline operating point, **seven in ten NT FPs pass**. `sweeps/p4_scoping.jl` (pool only, CV out-of-fold
scores of the "+ ephemeris" catalog model; `sweeps/results/p4_scoping.log`) shows what the 173 missed ones are:
- **Low SNR, but not single events:** median SNR ~8 (planets ~20). Usually 6–100 observed transits, and periods of
  10–200 d (35–40% missed there, vs 4% below 1.5 d).
- **U-shaped, not V-shaped** (median V-shape 0.25, planets 0.45), so shape alone does not flag them.
- **Which robovetter tests caught them** (its comments, read for analysis only, never inputs):

| Robovetter test (comment) | Share of our NT misses | Missed within that test's NT FPs | What it measures |
|---|---|---|---|
| `MOD_NONUNIQ_ALT` | 24% | 40% | the event is not unique: other dips or bumps in the phased light curve are as significant (model-shift test, alternative detrending) |
| `CENT_FEW_DIFFS` | 20% | 23% | (centroid test could not run; not a cause) |
| `INCONSISTENT_TRANS`, `INDIV_TRANS_*` | ~17% | 18–41% | individual transits disagree; `SKYE`: transits pile up on days when many other stars show events too (rolling band, spacecraft events) |
| `LPP_ALT_TOO_HIGH` | 5% | 24% | shape far from known transits (dimensionality-reduced shape metric) |

So the misses are mostly **events that are not unique in their own light curve, or whose individual transits
are inconsistent or fall on bad times**. Our Phase 2 features only partly cover these. `lc_sec_sig` tests one
secondary dip, `lc_timing` tests timing scatter and `lc_sev_ratio` tests single-event dominance. None compares the
primary against all other dips *and bumps*, tests per-transit depth consistency, or checks the transit times against
other stars.

## 2. Candidate features (all label-free, measured by us)

| # | Feature group | Data | Cost (estimate) |
|---|---|---|---|
| N-a | **Uniqueness (model-shift style):** significance of the primary vs the strongest other dip and the strongest bump at any phase, and vs the red-noise level (as Coughlin et al. 2016, MS1–MS3) | our cached light curves (14.3 GB) | a `p2_features`-style pass, ~1–1.5 h at 2 threads, ~2–3 GB |
| N-b | **Per-transit consistency:** χ² of individual transit depths about the mean, the fraction of transits with SNR below 0 | cached light curves | same pass as N-a |
| N-c | **Bad-time clustering (SKYE-like):** for each KOI, the fraction of its transits on days when an excess of *other stars'* DR25 TCE transits fall (rolling band, thermal events, momentum dumps) | DR25 TCE ephemerides (already downloaded) and our transit times | minutes, < 2 GB |
| N-d | **Alternative detrending:** our SNR again with a shorter and a longer biweight window; a signal that vanishes under one is a systematic | cached light curves | ~1.5 h (re-detrend), ~2–3 GB |

The CNN route (shape learning) was tried in Phase 2 and added nothing, so it is not proposed again.

## 3. Decisions (approved 11/13: the recommendation in every row)

| # | Decision | Chosen (= recommendation) | Why |
|---|---|---|---|
| N1 | Phase 4 target | **NT**, as above | The largest error group left; 70% pass at 98% recall |
| N2 | Feature groups | **N-c first** (cheap, catalog-only), **then N-a + N-b in one light-curve pass**; N-d only if those leave a gap | Ordered by cost and by the share of misses each targets |
| N3 | Family | **Into both existing families** (catalog and + centroid) as the new Phase 4 baseline, unlike Phase 3's separate family | These are our own measurements of the target's light curve, not re-implementations of a robovetter flag; the same status as the Phase 2 LC features. N-c uses other stars' *times* only (no labels), like Phase 3 |
| N4 | Base model | **Phase 2a features + the new ones**, compared paired against Phase 2a. The "+ ephemeris" models, if adopted, get the same addition as a second comparison | Keeps one clean lineage; ephemeris stays a separate family |
| N5 | Evaluation | **CV only** (the test set is spent), the same protocol and blind-spot table. Success: NT-only missed at 98% recall falls from ~70% by at least 10 points with the CI excluding 0; no AUC or CONFIRMED-lost regression | Same style as Phase 2/3 criteria |

## 4. Milestones (proposed)

| M | What | Compute |
|---|---|---|
| M0 | N-c: per-day event-excess histogram from the DR25 TCE list and the KOI table (label-free), per-KOI bad-time fraction; label-free sanity (does it flag the known Kepler rolling-band/thermal periods?) | minutes, < 2 GB |
| M1 | N-a + N-b in `src/LCFeatures.jl` (new functions), one pass over the cache; pool-only per-class look | ~1.5 h, 2–3 GB |
| M2 | CV of the new features in both families (4 runs of 5 × 10), paired vs Phase 2a; blind-spot table, menu, low-FP-rate recall vs `koi_score` | ~15–20 min, < 2 GB |
| M3 | Write-up notebook, README/STATUS; with the user: adopt or not | ~½ day |

## 4.1 Progress

- **11/13, Phase 3 freeze:** `models/p3_frozen.jld2` (`sweeps/p3_freeze.jl`), post-test, CV-validated.
- **11/13, M0 (N-c) done** (`src/BadTimes.jl`, `sweeps/p4_badtimes.jl`; `sweeps/results/p4_badtimes*.log/.csv`).
  Sources: the 29,180 DR25 TCEs with P ≥ 1 d. Predicted transit midpoints are counted in 0.25-d bins (3-bin
  smoothed). Features: global robust z, with each TCE weighted 1/N_transits and the target's own star removed, and
  a local Poisson significance from TCEs on other stars within 30′. Both are given as a mean over the KOI's transits
  and as the fraction of its transits in bad bins.
  - Label-free check: unweighted, the global series is swamped by short-period TCEs and shows nothing. Weighted, 77
    windows have z > 4 (22 with z > 8; worst at BKJD 1093.9, 1339.9, 601.6, 972.1, 1561.1) vs 0–5 in random-epoch
    nulls. Locally, 905 KOIs have a transit in a bad bin vs ~60 in the null.
  - Pool look (labels for analysis only): the signal is real but small. Only 99 pool FPs carry the robovetter's SKYE
    comment, and the missed NT-only FPs are not concentrated in bad times; if anything they sit there less than the
    caught ones. Single-feature AUC, NT-only vs planets: 0.57–0.58 (global), ~0.51 (local). So N-c alone will not
    close the gap. It still goes into M2 together with N-a + N-b, as planned.
- **11/13–14, M1 (N-a + N-b) done** (`LC_NT_FEATURES` in `src/LCFeatures.jl`, `sweeps/p4_ntfeatures.jl`; 33 min,
  2.1 GB, 9,271 KOIs, no errors). The features are:
  - N-a: `nt_red` (red-noise factor), `nt_pos_sig` / `nt_ter_sig` (strongest off-transit bump / tertiary dip) and
    `nt_ms_ter` / `nt_ms_pos` (primary minus those).
  - N-b: `nt_chi2` (per-transit depth χ²), `nt_negfrac`, `nt_chases`, `nt_rubble`. Chases and rubble are cheap extra
    per-transit checks of the same kind as the robovetter's INDIV_TRANS tests.
  - Pool look: the uniqueness features separate the missed NT FPs from planets (single-feature AUC 0.89 for
    `nt_ms_ter`); the per-transit ones barely do (0.50–0.57).
- **11/14, M2 done** (`sweeps/p4_cv.jl`, 8 runs, 34 min, < 2 GB; `sweeps/p4_m2.jl [--eph]`,
  `sweeps/results/p4_m2*_*.csv`). **Negative for NT; criterion N5 not met.**
  - vs Phase 2a: DR25 AUC 0.9705 → 0.9736 (+0.0031 [+0.0019, +0.0043]) catalog, 0.9869 → 0.9878 (+0.0008 [+0.0003,
    +0.0013]) + centroid.
  - That gain comes from **contamination**, not NT: EC-only misses at p = 0.5 fall 18.0 → 9.7% and 23.2 → 15.7%.
    NT-only misses do not move: 70.7 → 70.3% and 60.2 → 60.8% at 98% recall, CIs spanning 0.
  - vs "+ ephemeris" (the adopted lineage): DR25 AUC +0.0006 [−0.0001, +0.0012] / +0.0002 [−0.0002, +0.0005]. The
    EC overlap is gone, NT misses are unchanged, and at 98% recall in the + centroid family they are slightly worse
    (59.7 → 61.9%, CI above 0). No CONFIRMED regression anywhere.
  - Why (`sweeps/p4_snrmatched.jl`, pool only): within `koi_model_snr` bins the new features do separate NT FPs
    from planets (`nt_ms_ter` AUC 0.75 / 0.86 / 0.93 at SNR 7–10 / 10–15 / 15–30). But the Phase 2 features already
    carry the same information (`lc_sev_ratio` 0.76 / 0.87 / 0.81, `lc_sec_sig` 0.65 / 0.80 / 0.63), so the models
    gain nothing new. The NT FPs that still pass are the ones none of these measurements tell apart from
    low-SNR planets.
  - Per N2, this leaves a gap, so **N-d (alternative detrending)**, the last planned feature group, runs next.
- **11/14, N-d done** (`LC_ND_FEATURES` in `src/LCFeatures.jl`, `sweeps/p4_ntfeatures.jl --nd`; 1.6 h, 2.1 GB,
  9,271 KOIs, no errors; CV `sweeps/p4_cv.jl ephnd_* centephnd_*`, `sweeps/p4_m2.jl --nd`,
  `sweeps/results/p4_m2nd_*.csv`). **Small gain in the catalog family only; criterion N5 not met.**
  - Features: box depth and SNR re-measured after a short unmasked biweight (window max(0.25 d, 2.5 durations),
    which eats a long smooth "transit") and a long masked one (max(2 d, 10 durations)), as ratios to the default
    detrending. Pool look: they barely separate NT FPs from planets (single-feature AUC 0.50–0.58, also within SNR
    bins); they mainly flag secondary-eclipse (SS) FPs, whose long-window SNR drops.
  - CV vs "+ ephemeris": DR25 AUC unchanged (+0.0000 [−0.0003, +0.0004] catalog, +0.0001 [−0.0000, +0.0003]
    + centroid).
  - Catalog family at 98% recall: NT-only misses 70.0 → 67.9% (−2.1 [−2.9, −1.3]; DR25 KOIs 54.1 → 52.6%),
    centroid-only misses −1.6 [−2.5, −0.8], all FPs kept 28.7 → 27.9% (−0.8 [−1.1, −0.6]). Real but far from the
    10-point target.
  - + centroid family: no change anywhere (NT 59.7 → 59.7%). No CONFIRMED regression in either family.
  - With N-d, every planned feature group (N-a to N-d) has been tried. Adoption is the user's call.
- **11/14, M3 write-up done:** `notebooks/exo111426.ipynb` (sources `sweeps/nbcells/exo111426.jl`), README Phase 4
  section.

## 5. Rules carried over

The test split is never re-scored. No provenance/DR25 inputs, no missing-indicator features, no robovetter outputs as
inputs; comments and flags only for grouping. Jobs at below-normal priority, ≤ 2 threads; tell the hub the RAM and
threads before heavy runs.
