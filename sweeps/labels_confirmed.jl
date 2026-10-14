# Post-test experiment (10/13/2026): train on CONFIRMED vs FALSE POSITIVE only, then score CANDIDATEs as an
# application. CANDIDATE labels are the noisiest (step 2: many "missed planets" are CANDIDATEs that look like FPs).
# Same final recipes as step 5 (quantile MLP, tuned trees; catalog-only and catalog + centroid), same standard folds,
# 5 shuffles x 10 seeds. train_filter drops CANDIDATEs from TRAINING only; every pool KOI still gets an OOF score,
# so the runs pair with the train-on-all references. CV only: the spent test set is not touched. Resumable per job.
#   julia -t 2 sweeps/labels_confirmed.jl [mlp] [trees] [cent_mlp] [cent_trees]
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

df_raw = load_koi()
df = clean_koi(df_raw)
pool0, _ = cv_pool(df)
pool = add_centroid_features(pool0, df_raw)
not_cand = pool.koi_disposition .!= "CANDIDATE"
const MLP_KW  = (lr = 3e-3, epochs = 600, patience = 50, transform = :quantile)
const TREE_KW = (trainer = train_evotree, max_depth = 8, min_weight = 10.0, L2 = 10.0, colsample = 0.8)
const CENT = [FEATURES; CENTROID_FEATURES]
jobs = isempty(ARGS) ? ["mlp", "trees", "cent_mlp", "cent_trees"] : ARGS
println("training KOIs: $(sum(not_cand)) of $(nrow(pool)) (CANDIDATEs dropped from training only)")

function cached(file, name; kw...)
    isfile(joinpath(CV_DIR, file)) && (println("loaded: ", file); return load_cv(file))
    cv = cv_run(pool; repeats = 5, seeds = 1:10, name, train_filter = not_cand, kw...)
    println("saved: ", save_cv(file, cv))
    return cv
end

"mlp" in jobs        && cached("cv_mlpq_conftrain_r5s10.jld2", "mlpq_conftrain"; MLP_KW...)
"trees" in jobs      && cached("cv_evotree_conftrain_r5s10.jld2", "evotree_conftrain"; TREE_KW...)
"cent_mlp" in jobs   && cached("cv_mlpq_centroid_conftrain_r5s10.jld2", "mlpq_centroid_conftrain"; features = CENT, MLP_KW...)
"cent_trees" in jobs && cached("cv_evotree_centroid_conftrain_r5s10.jld2", "evotree_centroid_conftrain"; features = CENT, TREE_KW...)
println("DONE")
