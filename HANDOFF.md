# Handoff: moving the project to the 8 GB laptop (10/21/2026)

Written on the desktop (DESKTOP-BBASC81) when the project moved to the user's second laptop (8 GB RAM, 1 TB disk).
Read this, then `CLAUDE.md`, `STATUS.md`, `PHASE2_PLAN.md` and `sweeps/REPORT.md`.

## 1. State at handoff
- Phase 1 is finished and frozen (`models/step5_frozen.jld2`). The sealed test split was used once (10/13) and is spent.
- Phase 2: D1–D7 approved (10/19). M0 pilot passed (`notebooks/exo101926.ipynb`). M1 download finished 10/20:
  7,927 / 7,927 stars, 0 errors, 14.3 GB cache.
- Next: M2 (detrend + views + QA for all KOIs), then M3 (Phase 2a light-curve features + the existing CV protocol).

## 2. Fresh-machine setup
1. Install Julia 1.12 (`winget install julia -s msstore` gives juliaup; the desktop runs 1.12.6) and Git.
2. `git clone https://github.com/mburchnmsu/exoplanet-vetting.git "Exoplanet vetting"`
3. In the repo: `julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"`. The first precompile
   (Flux, CairoMakie, EvoTrees, FITSIO) takes ~10–20 min. FITSIO brings its own CFITSIO binary, so nothing else needs
   installing. `src/VettingSetup.jl` also runs `Pkg.instantiate()` itself.
4. Smoke test: `julia --project=. -e "include(\"src/VettingSetup.jl\"); df = clean_koi(load_koi()); println(size(df))"`
   should print `(9201, 15)`. The KOI table `data/koi_cumulative.csv` is in git, so no download is needed.
5. Notebooks: `sweeps/tools/run_nb.jl` builds and runs them headless with no Jupyter needed. To run them inside VS Code
   or Jupyter, add IJulia to the global environment (`julia -e "using Pkg; Pkg.add(\"IJulia\")"`); kernel name
   `julia-1.12`.

## 3. Git-ignored data the laptop needs
Only `data/lightcurves/` (everything else, including `models/` and `sweeps/results/`, is in git):

| Path | Size | Needed for | How to recreate |
|---|---|---|---|
| `data/lightcurves/cache/` | 14.3 GB, 7,927 files | Everything in M2 onwards | `sweeps/p2_download.jl` (~2.3 h, resumable) |
| `data/lightcurves/manifest.csv` | 0.5 MB | The download's resume logic and download stats | Written by the download |
| `data/lightcurves/pilot_views.jld2` | 2 MB | Notebook `exo101926` only | `sweeps/p2_pilot.jl` (downloads are skipped when cached; ~1–2 min) |

**Recommendation: copy the whole `data/lightcurves/` folder** (external drive or network share; `cache/`,
`manifest.csv` and `pilot_views.jld2` together, skipping any `tmp/`). It is faster than 2.3 h, gives identical
inputs, and spares MAST. If no drive is handy, re-running the download is fine: it is network-bound and uses < 1 GB
RAM. Copy the manifest together with the cache. Without it, the download treats cached stars as new and writes rows
with zero sizes.

## 4. Standing rules (from the user; most are also in `CLAUDE.md` / `sweeps/REPORT.md`)
- **Sealed test:** never re-score it (`sweeps/step5_test.jl` refuses to rerun). Phase 2 gets exactly one final test
  scoring (D5). Judge everything else by CV only (`compare_cv`, paired star bootstrap; the CI must exclude 0).
- **Inputs:** never DR25/provenance as a feature (`is_dr25` is for selecting and reporting rows only). No
  missing-indicator features. Never robovetter flags or `koi_score`. Centroid information only in the separate
  "+ centroid" family.
- **Reporting:** the headline is on DR25 KOIs, with supplemental KOIs reported separately. The headline operating point
  is **98% planet recall**; also show best F1, ROC-AUC and the full threshold menu.
- **Workflow:** pull at the start and commit + push at the end. Keep `STATUS.md` current for the hub (format in
  `CLAUDE.md`). Send a push notification on major progress. Dated notebooks go flat in `notebooks/` with the
  `notebooks/README.md` index (sources in `sweeps/nbcells/`). In notebooks the user edited, change only path lines.
  Ask the user before acting on open design decisions. Keep this project separate from zodi and solar.

## 5. Adjustments for 8 GB RAM
- **Claude Code stops its own background jobs when the whole machine is low on memory** (this happened twice on
  the desktop). On 8 GB this is likelier, so launch multi-hour runs from a normal terminal, or start Claude Code with
  `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`. All long scripts are resumable.
- One Julia session with Flux + CairoMakie loaded is ~1.5–2 GB. Avoid running a sweep and open notebook kernels at the
  same time, and close unused VS Code Julia kernels.
- **M2 (views):** memory per star is small (~50–100 MB). Process stars in a loop, save views as **Float32** in shards
  (e.g. 500 stars each) so the run is resumable. All views for ~9,200 KOIs are ~100–150 MB as Float32. On the desktop
  this was ~1.4 s per star, ~3 h single-threaded; use 2 threads over stars on the laptop and expect it to be slower.
- **M3 (features + trees/MLP CV):** the same footprint as Phase 1 (~2 GB). Fine. Use `-t 2` to `-t 4`.
- **M4 (CNNs):** RAM is not the limit (views fit in ~150 MB; a < 200k-parameter model trains in ~1–2 GB); the CPU
  is. Keep the plan's timing gate (one epoch first). Be ready to use the 1001-bin global view and trim the D6
  budget (final comparison 5 shuffles × 3 seeds) if one model takes > ~10 min. Run one training process at a time.
- The priority wrapper used on the desktop (`cmd /c "start /b /wait /belownormal julia --project=. -t 2 ..."`) only
  mattered to share the PC with zodi. On the laptop exoplanet runs alone, so more threads are fine, and below-normal
  still keeps the laptop usable.
