# Step 4, part 2: feature ablation and deeper trees, against the new references
#   MLP   = lr3_e600 + transform = :quantile    (cv_mlp_lr3_e600_quantile_r5s10.jld2)
#   trees = tuned EvoTrees, depth 8             (cv_evotree_tuned_r5s10.jld2)
# (a) koi_count ablation: both models without koi_count, standard folds (5 x 10)
# (b) deeper trees: depth 10 and 12 screened on the tuning folds (fold_seed0 = 5000, 2 x 3) like the step-3 grid;
#     the best goes to the standard folds only if it beats depth 8 there by > 0.0005
# koi_insol is NOT tested: it is a perfect rank copy of koi_teq (Spearman 1.0000 on the pool), so it adds exactly
# zero information for trees and for the quantile-transformed MLP.
# Resumable. Usage:  julia -t auto sweeps/step4_features.jl [nocount_mlp] [nocount_trees] [deep_trees]
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

df = clean_koi(load_koi())
pool, _ = cv_pool(df)
y = pool.label
const MLP_KW  = (lr = 3e-3, epochs = 600, patience = 50, transform = :quantile)
const TREE_KW = (trainer = train_evotree, max_depth = 8, min_weight = 10.0, L2 = 10.0, colsample = 0.8)
const NO_COUNT = filter(!=(:koi_count), FEATURES)
jobs = isempty(ARGS) ? ["nocount_mlp", "nocount_trees", "deep_trees"] : ARGS

function cached(file, name; kw...)
    isfile(joinpath(CV_DIR, file)) && (println("loaded: ", file); return load_cv(file))
    cv = cv_run(pool; name, kw...)
    println("saved: ", save_cv(file, cv))
    return cv
end

if "nocount_mlp" in jobs
    cached("cv_mlpq_nocount_r5s10.jld2", "mlpq_nocount"; repeats = 5, seeds = 1:10, features = NO_COUNT, MLP_KW...)
end
if "nocount_trees" in jobs
    cached("cv_evotree_nocount_r5s10.jld2", "evotree_nocount"; repeats = 5, seeds = 1:10, features = NO_COUNT, TREE_KW...)
end
if "deep_trees" in jobs
    grid = CSV.read(joinpath(CV_DIR, "evotree_grid.csv"), DataFrame)
    ref = grid.roc_auc[(grid.max_depth .== 8) .& (grid.min_weight .== 10) .& (grid.L2 .== 10) .& (grid.colsample .== 0.8)][1]
    rows = NamedTuple[(max_depth = 8, roc_auc = ref, logloss = NaN, median_rounds = NaN)]
    for d in (10, 12)
        cv = cached(joinpath("mlp_screen", "treescreen_depth$(d).jld2"), "tree_d$d"; repeats = 2, seeds = 1:3,
                    fold_seed0 = 5000, verbose = false, TREE_KW..., max_depth = d)
        s = oof_scores(cv)
        push!(rows, (max_depth = d, roc_auc = fast_auc(y, s), logloss = logloss(y, s),
                     median_rounds = median(cv.results.best_epoch)))
        @printf("tree depth %2d (tuning folds): AUC %.4f  logloss %.4f  rounds %.0f\n", d, rows[end].roc_auc, rows[end].logloss, rows[end].median_rounds)
    end
    deep = DataFrame(rows)
    CSV.write(joinpath(CV_DIR, "evotree_depth_screen.csv"), deep)
    best = deep[argmax(deep.roc_auc), :]
    if best.max_depth != 8 && best.roc_auc - ref > 0.0005
        println("depth $(best.max_depth) beats depth 8 by $(round(best.roc_auc - ref; digits = 4)): running standard folds")
        cached("cv_evotree_depth$(best.max_depth)_r5s10.jld2", "evotree_d$(best.max_depth)";
               repeats = 5, seeds = 1:10, TREE_KW..., max_depth = best.max_depth)
    else
        println("no depth beats depth 8 by > 0.0005 on the tuning folds: depth 8 stays")
    end
end
println("DONE")
