# Phase 3, M2 part 1: CV of the two "+ ephemeris" models (PHASE3_PLAN.md, E2). The Phase 2a recipes with the
# ephemeris-matching features added, on the standard folds (5 shuffles x 10 seeds), so every run is paired with its
# Phase 2a counterpart:
#   eph_mlp / eph_trees         : FEATURES + LC_FEATURES + EPHEMERIS_FEATURES        vs cv_mlpq_lc / cv_evotree_lc
#   centeph_mlp / centeph_trees : FEATURES + CENTROID_FEATURES + LC_FEATURES + LC_CENTROID_FEATURES + EPHEMERIS_FEATURES
#                                                                                    vs cv_mlpq_lccent / cv_evotree_lccent
# Non-finite features -> training median (prepare_data; no missing indicators). Resumable per job.
# Needs sweeps/results/p2_lcfeatures.csv and p3_ephemeris.csv (sweeps/p3_match.jl). Analysis: sweeps/p3_m2.jl.
#   --eb : v2, the same features with the Kepler EB catalog added to the matching (p3_ephemeris_eb.csv) ->
#          cv_*_ephEB / cv_*_centephEB, still paired vs Phase 2a here (v2 vs v1 in p3_m2.jl --eb)
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p3_cv.jl [--eb] [jobs...]"
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
using Dates

const EB = "--eb" in ARGS
const SFX = EB ? "EB" : ""
df_raw = load_koi()
pool0, _ = cv_pool(clean_koi(df_raw))
pool = add_centroid_features(pool0, df_raw)
pool = leftjoin(pool, CSV.read(joinpath(CV_DIR, "p2_lcfeatures.csv"), DataFrame); on = :kepoi_name, order = :left)
eph = CSV.read(joinpath(CV_DIR, EB ? "p3_ephemeris_eb.csv" : "p3_ephemeris.csv"), DataFrame)
pool = leftjoin(pool, eph[:, [:kepoi_name; EPHEMERIS_FEATURES]]; on = :kepoi_name, order = :left)
@assert pool.kepoi_name == pool0.kepoi_name
for c in [LC_FEATURES; LC_CENTROID_FEATURES; EPHEMERIS_FEATURES]
    pool[!, c] = Float64.(coalesce.(pool[!, c], NaN))
end
dr = is_dr25(pool, df_raw)
const MLP_KW  = (lr = 3e-3, epochs = 600, patience = 50, transform = :quantile)
const TREE_KW = (trainer = train_evotree, max_depth = 8, min_weight = 10.0, L2 = 10.0, colsample = 0.8)
const EF  = [FEATURES; LC_FEATURES; EPHEMERIS_FEATURES]
const ECF = [FEATURES; CENTROID_FEATURES; LC_FEATURES; LC_CENTROID_FEATURES; EPHEMERIS_FEATURES]
jobs = filter(!=("--eb"), ARGS)
jobs = isempty(jobs) ? ["eph_trees", "eph_mlp", "centeph_trees", "centeph_mlp"] : jobs

function cached(file, name; kw...)
    isfile(joinpath(CV_DIR, file)) && (println("loaded: ", file); return load_cv(file))
    t0 = time()
    cv = cv_run(pool; repeats = 5, seeds = 1:10, name, kw...)
    @printf("%s  saved: %s  (%.1f min)\n", Dates.format(now(), "mm-dd HH:MM"), save_cv(file, cv), (time() - t0) / 60)
    flush(stdout)
    return cv
end

specs = [("eph_trees", "cv_evotree_eph$(SFX)_r5s10.jld2", "cv_evotree_lc_r5s10.jld2", (features = EF, TREE_KW...)),
         ("eph_mlp", "cv_mlpq_eph$(SFX)_r5s10.jld2", "cv_mlpq_lc_r5s10.jld2", (features = EF, MLP_KW...)),
         ("centeph_trees", "cv_evotree_centeph$(SFX)_r5s10.jld2", "cv_evotree_lccent_r5s10.jld2", (features = ECF, TREE_KW...)),
         ("centeph_mlp", "cv_mlpq_centeph$(SFX)_r5s10.jld2", "cv_mlpq_lccent_r5s10.jld2", (features = ECF, MLP_KW...))]

"Restrict a CV result to a subset of pool rows (for the DR25 headline)."
sub(cv, m) = (results = cv.results, oof = cv.oof[m, :, :], folds = cv.folds[m, :], config = cv.config, kepoi = cv.kepoi[m])

for (job, file, base, kw) in specs
    job in jobs || continue
    cv = cached(file, job; kw...)
    b = load_cv(base)
    for (lab, m) in [("all pool", trues(nrow(pool))), ("DR25", dr)]
        c = compare_cv(sub(cv, m), sub(b, m), pool[m, :]; B = 500)
        @printf("  %-13s %-8s AUC %.4f vs Phase 2a %.4f: Δ %+.4f [%+.4f, %+.4f]\n", job, lab, c.A, c.B, c.delta,
                c.ci_lo, c.ci_hi)
    end
    flush(stdout)
end
println(Dates.format(now(), "mm-dd HH:MM"), "  DONE")
