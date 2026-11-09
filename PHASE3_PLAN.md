# Phase 3 plan: catching contamination false positives by ephemeris matching

*Drafted 11/02/2026, after Phase 2 was finished and tested. The user chose (11/02):*
- *start with the contamination (EC) blind spot;*
- *put the matching features in a **separate family**;*
- *match against the KOI table, the Kepler eclipsing-binary catalog and the DR25 TCE list.*

*The "not transit-like" (NT) blind spot may follow later. TESS would be its own phase.*

## 1. Goal and what counts as success

Phase 2 left contamination as one of the two largest error groups. Many EC false positives are another star's
signal leaking into the target's pixels:
- direct overlap of the stars' images;
- charge bleeding along a CCD column;
- electronic cross-talk between CCD channels.

Kepler's robovetter finds these by **ephemeris matching**: a KOI whose period and transit time match a signal on
another star is flagged EC. No per-KOI feature we have can see this, because the evidence lives on *another* star.

**Targets (Phase 2a, CV, DR25 KOIs; nested hard calls, mean over fold shuffles):**

| Group | Catalog family | + centroid family |
|---|---|---|
| EC-only FPs missed at p = 0.5 (n = 111) | 18.0% | 23.2% |
| EC-only FPs missed at 98% recall | 44.3% | 33.7% |
| DR25 ROC-AUC | 0.971 | 0.987 |

**Success criteria (judged by CV only, with the existing paired protocol; the test set is spent, see §4):**
1. **Main:** EC-only FPs missed at p = 0.5 fall below 15% in at least one "+ ephemeris" model, with the paired
   star-bootstrap CI on the drop excluding 0. Also reported for every EC-flagged FP, not only EC-only.
2. **No regressions:**
   - DR25 ROC-AUC is not lower than the matching Phase 2a model (CI);
   - CONFIRMED planets lost at 98% recall do not rise.
3. **Low-FP-rate target (reported every time):** recall at 5% and 10% FP rate on DR25 KOIs vs the robovetter's
   `koi_score`, where the robovetter still wins in the catalog family.

Honest expectation: EC-only FPs are 111 of 7,199 DR25 pool KOIs (1,148 KOIs carry the EC flag in total), so the
overall AUC will move little. The judgement rests on the EC miss rates, and their CIs will be wide.

## 2. Decisions

| # | **DECISION** | Status / recommendation | Why |
|---|---|---|---|
| E1 | First Phase 3 target | **EC first** (user, 11/02) | Cheap (catalog data only), targets a known weakness |
| E2 | Where the features go | **Separate "+ ephemeris" family** (user, 11/02). Recommended form: two new models, "catalog + ephemeris" and "catalog + centroid + ephemeris" (each = the Phase 2a features + the matching features). The two Phase 2 families stay as they are | The EC flag is itself an ephemeris-matching test, so these features are partly circular, like the centroids. Keeping them out of the existing families keeps those clean |
| E3 | Catalogs to match against | **KOI table + Kepler EB catalog + DR25 TCE list** (user, 11/02) | Contamination sources are often EBs, or signals that never became KOIs |
| E4 | What may be used from the other catalogs | **Only ephemeris, shape and position:** period, epoch, duration, depth, KIC ID, RA/Dec, Kepler magnitude. **Never** the other objects' dispositions, robovetter flags, scores or comments | Label-free. Being *listed* in the EB catalog is information about the other star, which is the kind of evidence Kepler itself used. It is never information about the target |
| E5 | Same-star matches | **Excluded:** a target is never matched to its own star (same KIC) | Otherwise "is this star in the EB catalog / the DR25 TCE list" would leak the target's own label and DR25 provenance (both forbidden) |
| E6 | CCD position (cross-talk, column anomaly) | **Not in v1:** the KOI table has no CCD channel/column. Matches at any separation are kept, with the separation as a feature | Cross-talk matches can be far apart on the sky. The CCD channel could be added later from the FITS headers if v1 shows promise |

## 3. Method

**Matching (new `src/Ephemeris.jl`).** For target A and every catalog entry B on another star, and small period
ratios m:n (m, n ≤ 10, reduced, so 1:1, 1:2, 2:1, 2:3, ...):
- **Period drift:** how far the two ephemerides drift apart over the mission, in units of A's duration:
  `|P_A − (m/n)·P_B| × (mission span / P_A) / duration_A`.
- **Epoch offset:** the phase offset of B's transits from A's at A's epoch, in A's durations.
- **Match score:** the two combined. The best (smallest) score per target and per catalog becomes a feature.

Features per target (best match per catalog where it makes sense; names provisional):

| Feature | Meaning |
|---|---|
| `eph_score_koi`, `eph_score_eb`, `eph_score_tce` | best match score against each catalog (missing → a large value, not an indicator) |
| `eph_sep` | sky separation of the best match [arcsec] |
| `eph_dmag` | Kepler magnitude of the best match − the target's (negative = brighter source) |
| `eph_ratio_simple` | 1 if the best match is 1:1, else the ratio's complexity (m + n) |
| `eph_depth_ok` | whether the matched signal, diluted by the flux ratio, could produce the target's depth |
| `eph_n_matches` | number of matches below a fixed score threshold within a fixed radius |

The thresholds and radius are fixed in M1, from label-free checks, before any CV.

**Null check (label-free).** Repeat the matching with every catalog's epochs shifted at random. That shows how many
"matches" happen by chance at each score, so the score threshold can be chosen without labels.

**Models and CV.** Same learners, hyperparameters and folds as Phase 2a (MLP + trees 50/50, 5 shuffles × 10 seeds),
paired against Phase 2a, with the M5 blind-spot table and operating-point menu.

### 3.1 M0–M1 outcome (11/04/2026): what changed after the label-free checks

- **Catalogs:** the DR25 TCE list (34,032 TCEs) and DR25 stellar magnitudes are downloaded (`sweeps/p3_download.jl`,
  `data/catalogs/`, git-ignored). **The Kepler EB catalog could not be fetched**: both CDS mirrors timed out, and the
  Villanova CSV export returns "page not found". v1 runs without it. (It was fetched on 11/08 from the CfA mirror
  and tested as v2; see 3.2.)
- **The random-epoch null changed the design** (`sweeps/p3_match.jl`, `sweeps/results/p3_match.log`):
  - With all 63 ratios m:n ≤ 10 and no distance limit, ~90% of KOIs "match" something at score < 1, and the null
    gives the same rate: pure chance.
  - The real signal sits at small separations, simple ratios and tight scores. At score < 0.25, matches within 10"
    are ~95% real, within 1' ~90%, and at 3–10' ~55%. Beyond 30' the "other" ratios are chance.
- **v1 features** (`src/Ephemeris.jl`, `EPHEMERIS_FEATURES`; ratios 1:1, 1:2, 2:1 only):
  - from the best match within 10′ (`NEAR` = 600″): `eph_score_near`, `eph_logsep`, `eph_logleak`, `eph_dmag`;
  - `eph_nmatch_near`, the number of near stars matched with score < 0.25 (`MATCH_CUT`);
  - `eph_score_far`, the best 1:1 score beyond 10′ (cross-talk candidates; only ~38% real at < 0.25, so the model
    weighs it).

  "No match" takes the search limit (score 3, separation 10′, leak 6), not a missing indicator.
- **Size of the real signal:** near matches at score < 0.25 number 1,400 KOIs vs 369 by chance, an excess of
  ~1,030 KOIs, close to the 1,148 KOIs carrying the EC flag.

**Pool-only class check (`--classes`, `sweeps/results/p3_match_classes.log`):**

| Pool group | n | Near match (score < 0.25) | Far 1:1 match | Of near matches: other star could cause the dip |
|---|---|---|---|---|
| Planets | 4,163 | 0.5% | 0.4% | 4.5% |
| EC-only FPs | 111 | **68.5%** | 46.8% | 69.7% |
| Any EC-flagged FP | 1,022 | **86.0%** | 30.0% | 69.5% |
| SS-only FPs | 1,307 | 22.3% | 2.8% | 2.1% |
| CO-only FPs | 464 | 2.6% | 1.1% | 58.3% |
| NT-only FPs | 849 | 3.4% | 2.0% | 10.7% |

- The features find most contamination FPs and almost no planets.
- The SS-only matches are mostly the **sources** (bright EBs that contaminate their neighbours): only 2% of them
  could have been caused by the matched star. The leak feature tells victim from source.
- Single-feature AUC, EC-only FPs vs planets: 0.90 (near score, separation, leak), 0.82 (far score).

### 3.2 M2 outcome (11/08/2026): criteria met

CV, 5 × 10, paired vs Phase 2a (`sweeps/p3_cv.jl`, 13 min for all four runs; analysis `sweeps/p3_m2.jl`,
`sweeps/results/p3_m2_*.csv`):

| | Catalog: Phase 2a → + ephemeris | + centroid: Phase 2a → + ephemeris |
|---|---|---|
| DR25 ROC-AUC (Δ [95% CI]) | 0.971 → **0.976** (+0.006 [+0.004, +0.008]) | 0.987 → **0.989** (+0.002 [+0.001, +0.003]) |
| EC-only FPs missed, p = 0.5 | 18.0 → **4.3%** (Δ −13.7 [−20.5, −7.1]) | 23.2 → **6.7%** (−16.6 [−23.3, −10.0]) |
| EC-only FPs missed, 98% recall | 44.3 → 15.0% | 33.7 → 13.3% |
| EC-flagged FPs missed, 98% recall | 22.4 → 3.2% | 9.3 → 2.3% |
| DR25 FPs kept at 98% recall | 25 → 20% | 14 → 12% |
| CONFIRMED lost at 98% recall | 0.4 → 0.4% | 0.3 → 0.4% (Δ 0.0 [−0.1, +0.2]) |
| Recall at 5% / 10% FP rate (`koi_score` 0.953 / 0.968) | 0.849 / 0.918 → 0.880 / 0.936 | 0.935 / 0.966 → 0.946 / **0.973** |

- §1 criteria: **1 (main) met** in both families. **2 met**: AUC up, CONFIRMED lost unchanged within CI.
  **3 reported**: + centroid + ephemeris beats `koi_score` at a 10% FP rate and below ~3%, and trails it at 5%.
- Side effect: in the catalog family, CO-only FPs missed rise 34.6 → 40.8% (p = 0.5; CI above 0). Their sources
  are mostly non-targets with no TCE, so "no match" reads as planet-like. In + centroid the rise is small
  (5.3 → 6.7%). NT-only +0.8 points.
- **v2 with the Kepler EB catalog** (CfA VizieR mirror, 11/08; `--eb` flags): ~15 more real near matches. Every
  change vs v1 is ≈ 0 (`p3_m2eb_*.csv`), because most EBs are already DR25 TCEs. **v1 stays.**
- Write-up: `notebooks/exo110826.ipynb`. Open, for the user: adopt (freeze, labelled "post-test, CV-validated") or
  not; whether NT is next.

## 4. Rules carried over

- **No test-set use.** The sealed test split (seed 42) has been scored once per phase. Phase 3 is judged on CV only.
  Any frozen Phase 3 model would be labelled "post-test, CV-validated".
- Headline on DR25 KOIs, supplemental KOIs reported separately, 98% planet recall as the headline operating point.
- No provenance/DR25 inputs, no missing-indicator features, no robovetter outputs as inputs.
- Jobs at below-normal priority, ≤ 2 threads. Tell the hub the RAM and threads before heavy runs.

## 5. Milestones

| M | What | Effort / compute |
|---|---|---|
| M0 | Download the Kepler EB catalog (~2,900 EBs) and the DR25 TCE table (~34k TCEs), the small CSVs, into `data/`. Document columns; add their forbidden columns to the leakage guard | < 1 h; MBs |
| M1 | `src/Ephemeris.jl`: matching, features, the random-epoch null; label-free sanity checks (matches per KOI, the chance-match rate vs score); fix the thresholds. Then a pool-only look at match rates per class | ~1 day; minutes of CPU, < 2 GB |
| M2 | CV of the two "+ ephemeris" models (4 runs of 5 × 10), paired vs Phase 2a; blind-spot table, menu, low-FP-rate recall vs `koi_score` | ~30 min at 2 threads (Phase 2a took 32 min for all four), ~2–3 GB |
| M3 | Write-up notebook + README/STATUS; with the user: adopt or not, and whether NT comes next | ~½ day |
