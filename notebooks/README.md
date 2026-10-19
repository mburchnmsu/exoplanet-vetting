# Notebooks

New to the project? Start with `../ExoOnboarding.ipynb` (project root): data, physics, method, results, open issues.

Dated notebooks (`exoMMDDYY.ipynb`), oldest first. Setup cells load the code with
`include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))`. The notebooks from 10/10 on are built from
`sweeps/nbcells/<name>.jl` with `sweeps/tools/run_nb.jl`; the write-up is `sweeps/REPORT.md`.

- `exo100926.ipynb` (10/09/2026): the original tabular MLP baseline. Star-grouped split, training history, validation ROC/PR and confusion matrix, and test-set metrics.
- `exo101026.ipynb` (10/10/2026): steps 1–2. Noise floor of the CV protocol (fold/seed spread, null differences) and error analysis (robovetter flags, binned errors, supplemental KOIs, `koi_score` comparison).
- `exo101326.ipynb` (10/13/2026): steps 3–5 and the test. Baselines (logistic regression, trees), MLP tuning, input transforms, feature tests, centroid model and DR25 scoping, calibration and the operating-point menu, the one-time sealed-test evaluation, and the post-test CONFIRMED-only experiment.
