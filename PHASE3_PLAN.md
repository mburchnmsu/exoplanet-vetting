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
