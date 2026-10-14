# Step 4, part 3 (user decisions 10/13): centroid features as a separate model; supplemental-KOI scoping.
#   cent_mlp / cent_trees : reference MLP / trees with FEATURES + CENTROID_FEATURES (missing -> training median)
#   dr25_mlp / dr25_trees : reference models trained on DR25-detected KOIs only (predictions still for every
#                           pool KOI, same folds), to compare with training on all KOIs, scored on DR25 KOIs
# All on the standard folds, 5 shuffles x 10 seeds. Resumable.
#   julia -t auto sweeps/step4_scoping.jl [cent_mlp] [cent_trees] [dr25_mlp] [dr25_trees]
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

df_raw = load_koi()
df = clean_koi(df_raw)
pool0, _ = cv_pool(df)
pool = add_centroid_features(pool0, df_raw)       # same rows/order; extra columns only
dr25 = is_dr25(pool, df_raw)
const MLP_KW  = (lr = 3e-3, epochs = 600, patience = 50, transform = :quantile)
const TREE_KW = (trainer = train_evotree, max_depth = 8, min_weight = 10.0, L2 = 10.0, colsample = 0.8)
const CENT = [FEATURES; CENTROID_FEATURES]
jobs = isempty(ARGS) ? ["cent_mlp", "cent_trees", "dr25_mlp", "dr25_trees"] : ARGS

function cached(file, name; kw...)
    isfile(joinpath(CV_DIR, file)) && (println("loaded: ", file); return load_cv(file))
    cv = cv_run(pool; repeats = 5, seeds = 1:10, name, kw...)
    println("saved: ", save_cv(file, cv))
    return cv
end

"cent_mlp" in jobs   && cached("cv_mlpq_centroid_r5s10.jld2", "mlpq_centroid"; features = CENT, MLP_KW...)
"cent_trees" in jobs && cached("cv_evotree_centroid_r5s10.jld2", "evotree_centroid"; features = CENT, TREE_KW...)
"dr25_mlp" in jobs   && cached("cv_mlpq_dr25train_r5s10.jld2", "mlpq_dr25train"; train_filter = dr25, MLP_KW...)
"dr25_trees" in jobs && cached("cv_evotree_dr25train_r5s10.jld2", "evotree_dr25train"; train_filter = dr25, TREE_KW...)
println("DONE")
