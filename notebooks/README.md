# Notebooks

New to the project? Start with `../ExoOnboarding.ipynb` (project root): data, physics, method, results, open issues.

Dated notebooks (`exoMMDDYY.ipynb`), oldest first. Setup cells load the code with
`include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))`. The notebooks from 10/10 on are built from
`sweeps/nbcells/<name>.jl` with `sweeps/tools/run_nb.jl`; the write-up is `sweeps/REPORT.md`.

- `exo100926.ipynb` (10/09/2026): the original tabular MLP baseline. Star-grouped split, training history, validation ROC/PR and confusion matrix, and test-set metrics.
- `exo101026.ipynb` (10/10/2026): steps 1–2. Noise floor of the CV protocol (fold/seed spread, null differences) and error analysis (robovetter flags, binned errors, supplemental KOIs, `koi_score` comparison).
- `exo101326.ipynb` (10/13/2026): steps 3–5 and the test. Baselines (logistic regression, trees), MLP tuning, input transforms, feature tests, centroid model and DR25 scoping, calibration and the operating-point menu, the one-time sealed-test evaluation, and the post-test CONFIRMED-only experiment.
- `exo101926.ipynb` (10/19/2026): Phase 2 M0 pilot. 49 pool stars' light curves: download size/time and the
  extrapolation to all stars, raw vs detrended light curves, global/local/odd-even/secondary views per class,
  average views, centroid shifts, QA columns, and the M0 findings.
- `exo102426.ipynb` (10/24/2026): Phase 2 M2. Views for all 9,271 KOIs: coverage, label-free QA flags (observed
  transits, depth ratio, timing scatter vs per-transit SNR), median views per class on the pool (local, secondary,
  global, odd − even, centroid, planets by duration), and the M2 gate.
- `exo103026.ipynb` (10/30/2026): Phase 2 write-up (M3–M6). The light-curve features per class and alone,
  Phase 1 vs Phase 2a CV, blind spots (NT, EC, ultra-short periods), the CNN variants (tried, not used),
  calibration and the operating-point menu, the one-time sealed-test results, and the Phase 2 conclusions.
