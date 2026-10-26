# Phase 2, M3 part 2: Phase 2a CV. The Phase 1 recipes (quantile MLP lr 3e-3 / 600 epochs / patience 50, and tuned
# EvoTrees depth 8) with the light-curve features added, on the standard folds (5 shuffles x 10 seeds), so every run
# is paired with its Phase 1 counterpart:
#   lc_mlp / lc_trees         : FEATURES + LC_FEATURES                                  vs cv_mlp_lr3_e600_quantile / cv_evotree_tuned
#   lccent_mlp / lccent_trees : FEATURES + CENTROID_FEATURES + LC_FEATURES + LC_CENTROID_FEATURES
#                                                                                       vs cv_mlpq_centroid / cv_evotree_centroid
# Non-finite LC features -> training median (as for the centroid features; no missing indicators).
# Resumable per job. Needs sweeps/results/p2_lcfeatures.csv (sweeps/p2_features.jl).
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p2_cv.jl [jobs...]"
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

df_raw = load_koi()
df = clean_koi(df_raw)
pool0, _ = cv_pool(df)
pool = add_centroid_features(pool0, df_raw)
lc = CSV.read(joinpath(CV_DIR, "p2_lcfeatures.csv"), DataFrame)
pool = leftjoin(pool, lc; on = :kepoi_name, order = :left)
@assert pool.kepoi_name == pool0.kepoi_name
for c in [LC_FEATURES; LC_CENTROID_FEATURES]
    pool[!, c] = Float64.(coalesce.(pool[!, c], NaN))
end
dr = is_dr25(pool, df_raw)
const MLP_KW  = (lr = 3e-3, epochs = 600, patience = 50, transform = :quantile)
const TREE_KW = (trainer = train_evotree, max_depth = 8, min_weight = 10.0, L2 = 10.0, colsample = 0.8)
const LCF  = [FEATURES; LC_FEATURES]
const LCCF = [FEATURES; CENTROID_FEATURES; LC_FEATURES; LC_CENTROID_FEATURES]
jobs = isempty(ARGS) ? ["lc_trees", "lc_mlp", "lccent_trees", "lccent_mlp"] : ARGS

function cached(file, name; kw...)
    isfile(joinpath(CV_DIR, file)) && (println("loaded: ", file); return load_cv(file))
    t0 = time()
    cv = cv_run(pool; repeats = 5, seeds = 1:10, name, kw...)
    @printf("saved: %s  (%.1f min)\n", save_cv(file, cv), (time() - t0) / 60)
    flush(stdout)
    return cv
end

specs = [("lc_trees", "cv_evotree_lc_r5s10.jld2", "cv_evotree_tuned_r5s10.jld2", (features = LCF, TREE_KW...)),
         ("lc_mlp", "cv_mlpq_lc_r5s10.jld2", "cv_mlp_lr3_e600_quantile_r5s10.jld2", (features = LCF, MLP_KW...)),
         ("lccent_trees", "cv_evotree_lccent_r5s10.jld2", "cv_evotree_centroid_r5s10.jld2", (features = LCCF, TREE_KW...)),
         ("lccent_mlp", "cv_mlpq_lccent_r5s10.jld2", "cv_mlpq_centroid_r5s10.jld2", (features = LCCF, MLP_KW...))]

"Restrict a CV result to a subset of pool rows (for the DR25 headline)."
sub(cv, m) = (results = cv.results, oof = cv.oof[m, :, :], folds = cv.folds[m, :], config = cv.config, kepoi = cv.kepoi[m])

for (job, file, base, kw) in specs
    job in jobs || continue
    cv = cached(file, job; kw...)
    b = load_cv(base)
    for (lab, m) in [("all pool", trues(nrow(pool))), ("DR25", dr)]
        c = compare_cv(sub(cv, m), sub(b, m), pool[m, :]; B = 500)
        @printf("  %-12s %-8s AUC %.4f vs Phase 1 %.4f: Δ %+.4f [%+.4f, %+.4f]\n", job, lab, c.A, c.B, c.delta,
                c.ci_lo, c.ci_hi)
    end
    flush(stdout)
end
println("DONE")
