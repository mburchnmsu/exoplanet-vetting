# Phase 4, M2 part 1: CV of the "+ NT" features (PHASE4_PLAN.md, N2–N5): bad-time clustering (BADTIME_FEATURES,
# sweeps/p4_badtimes.jl) and uniqueness / per-transit consistency (LC_NT_FEATURES, sweeps/p4_ntfeatures.jl), added
# to both families (N3). The Phase 2a / Phase 3 recipes on the standard folds (5 shuffles x 10 seeds), so every
# run is paired with its counterpart (N4):
#   nt_*        : Phase 2a catalog + NT                    vs cv_*_lc
#   centnt_*    : Phase 2a + centroid + NT                 vs cv_*_lccent
#   ephnt_*     : "+ ephemeris" catalog + NT               vs cv_*_eph       (the adopted Phase 3 lineage)
#   centephnt_* : "+ ephemeris" + centroid + NT            vs cv_*_centeph
# Non-finite features -> training median (prepare_data; no missing indicators). Resumable per job.
# Analysis: sweeps/p4_m2.jl.
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p4_cv.jl [jobs...]"
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
using Dates

df_raw = load_koi()
pool0, _ = cv_pool(clean_koi(df_raw))
pool = add_centroid_features(pool0, df_raw)
pool = leftjoin(pool, CSV.read(joinpath(CV_DIR, "p2_lcfeatures.csv"), DataFrame); on = :kepoi_name, order = :left)
eph = CSV.read(joinpath(CV_DIR, "p3_ephemeris.csv"), DataFrame)
pool = leftjoin(pool, eph[:, [:kepoi_name; EPHEMERIS_FEATURES]]; on = :kepoi_name, order = :left)
bt = CSV.read(joinpath(CV_DIR, "p4_badtimes.csv"), DataFrame)
pool = leftjoin(pool, bt[:, [:kepoi_name; BADTIME_FEATURES]]; on = :kepoi_name, order = :left)
ntf = CSV.read(joinpath(CV_DIR, "p4_ntfeatures.csv"), DataFrame)
pool = leftjoin(pool, ntf[:, [:kepoi_name; LC_NT_FEATURES]]; on = :kepoi_name, order = :left)
const NDFILE = joinpath(CV_DIR, "p4_ndfeatures.csv")              # N-d (sweeps/p4_ntfeatures.jl --nd)
isfile(NDFILE) && (pool = leftjoin(pool, CSV.read(NDFILE, DataFrame); on = :kepoi_name, order = :left))
@assert pool.kepoi_name == pool0.kepoi_name
for c in [LC_FEATURES; LC_CENTROID_FEATURES; EPHEMERIS_FEATURES; BADTIME_FEATURES; LC_NT_FEATURES;
          (isfile(NDFILE) ? LC_ND_FEATURES : Symbol[])]
    pool[!, c] = Float64.(coalesce.(pool[!, c], NaN))
end
dr = is_dr25(pool, df_raw)
const MLP_KW  = (lr = 3e-3, epochs = 600, patience = 50, transform = :quantile)
const TREE_KW = (trainer = train_evotree, max_depth = 8, min_weight = 10.0, L2 = 10.0, colsample = 0.8)
const NT  = [BADTIME_FEATURES; LC_NT_FEATURES]
const LF  = [FEATURES; LC_FEATURES]
const LCF = [FEATURES; CENTROID_FEATURES; LC_FEATURES; LC_CENTROID_FEATURES]
jobs = isempty(ARGS) ? ["nt_trees", "nt_mlp", "centnt_trees", "centnt_mlp",
                        "ephnt_trees", "ephnt_mlp", "centephnt_trees", "centephnt_mlp"] : ARGS

function cached(file, name; kw...)
    isfile(joinpath(CV_DIR, file)) && (println("loaded: ", file); return load_cv(file))
    t0 = time()
    cv = cv_run(pool; repeats = 5, seeds = 1:10, name, kw...)
    @printf("%s  saved: %s  (%.1f min)\n", Dates.format(now(), "mm-dd HH:MM"), save_cv(file, cv), (time() - t0) / 60)
    flush(stdout)
    return cv
end

specs = [("nt_trees", "cv_evotree_nt_r5s10.jld2", "cv_evotree_lc_r5s10.jld2", (features = [LF; NT], TREE_KW...)),
         ("nt_mlp", "cv_mlpq_nt_r5s10.jld2", "cv_mlpq_lc_r5s10.jld2", (features = [LF; NT], MLP_KW...)),
         ("centnt_trees", "cv_evotree_centnt_r5s10.jld2", "cv_evotree_lccent_r5s10.jld2", (features = [LCF; NT], TREE_KW...)),
         ("centnt_mlp", "cv_mlpq_centnt_r5s10.jld2", "cv_mlpq_lccent_r5s10.jld2", (features = [LCF; NT], MLP_KW...)),
         ("ephnt_trees", "cv_evotree_ephnt_r5s10.jld2", "cv_evotree_eph_r5s10.jld2",
          (features = [LF; EPHEMERIS_FEATURES; NT], TREE_KW...)),
         ("ephnt_mlp", "cv_mlpq_ephnt_r5s10.jld2", "cv_mlpq_eph_r5s10.jld2", (features = [LF; EPHEMERIS_FEATURES; NT], MLP_KW...)),
         ("centephnt_trees", "cv_evotree_centephnt_r5s10.jld2", "cv_evotree_centeph_r5s10.jld2",
          (features = [LCF; EPHEMERIS_FEATURES; NT], TREE_KW...)),
         ("centephnt_mlp", "cv_mlpq_centephnt_r5s10.jld2", "cv_mlpq_centeph_r5s10.jld2",
          (features = [LCF; EPHEMERIS_FEATURES; NT], MLP_KW...)),
         # N-d (alternative detrending), on the adopted "+ ephemeris" lineage only; run by name:
         #   p4_cv.jl ephnd_trees ephnd_mlp centephnd_trees centephnd_mlp
         ("ephnd_trees", "cv_evotree_ephnd_r5s10.jld2", "cv_evotree_eph_r5s10.jld2",
          (features = [LF; EPHEMERIS_FEATURES; LC_ND_FEATURES], TREE_KW...)),
         ("ephnd_mlp", "cv_mlpq_ephnd_r5s10.jld2", "cv_mlpq_eph_r5s10.jld2",
          (features = [LF; EPHEMERIS_FEATURES; LC_ND_FEATURES], MLP_KW...)),
         ("centephnd_trees", "cv_evotree_centephnd_r5s10.jld2", "cv_evotree_centeph_r5s10.jld2",
          (features = [LCF; EPHEMERIS_FEATURES; LC_ND_FEATURES], TREE_KW...)),
         ("centephnd_mlp", "cv_mlpq_centephnd_r5s10.jld2", "cv_mlpq_centeph_r5s10.jld2",
          (features = [LCF; EPHEMERIS_FEATURES; LC_ND_FEATURES], MLP_KW...))]

"Restrict a CV result to a subset of pool rows (for the DR25 headline)."
sub(cv, m) = (results = cv.results, oof = cv.oof[m, :, :], folds = cv.folds[m, :], config = cv.config, kepoi = cv.kepoi[m])

for (job, file, base, kw) in specs
    job in jobs || continue
    cv = cached(file, job; kw...)
    b = load_cv(base)
    for (lab, m) in [("all pool", trues(nrow(pool))), ("DR25", dr)]
        c = compare_cv(sub(cv, m), sub(b, m), pool[m, :]; B = 500)
        @printf("  %-15s %-8s AUC %.4f vs base %.4f: Δ %+.4f [%+.4f, %+.4f]\n", job, lab, c.A, c.B, c.delta,
                c.ci_lo, c.ci_hi)
    end
    flush(stdout)
end
println(Dates.format(now(), "mm-dd HH:MM"), "  DONE")
