# ==============================================================================
# BASELINES (step 3): models to measure the MLP against, on the same CV folds.
#  - Logistic regression: train_model(data; hidden = ()) (a single Dense layer,
#    same scaling, loss and early stopping as the MLP). Nothing extra needed here.
#  - Gradient-boosted trees (EvoTrees.jl): train_evotree, below.
# Both plug into cv_run via `trainer = ...` and are compared with compare_cv.
# Included automatically by VettingSetup.jl.
# ==============================================================================

import EvoTrees

"A fitted EvoTrees booster, truncated to the best round on the inner validation set."
struct TreeVetter
    model::EvoTrees.EvoTree
    ntree::Int          # trees used for prediction (bias tree + best_iter rounds)
end

predict_proba(t::TreeVetter, X::AbstractMatrix) =
    EvoTrees.predict(t.model, permutedims(Float64.(X)); ntree_limit = t.ntree)

"""
    train_evotree(data; seed = 1, nrounds = 3000, eta = 0.05, max_depth = 5,
                  min_weight = 5, rowsample = 0.8, colsample = 0.8, L2 = 1,
                  lambda = 0, nbins = 64, early_stopping_rounds = 100,
                  pos_weight = 1)

Gradient-boosted trees with logloss, early-stopped on the inner validation set
(the same inner split the MLP uses for early stopping). Trees ignore the
z-scaling and log transforms (both monotone), so they see the same information
as the MLP. `pos_weight` up-weights planets, like the MLP's weighted BCE.
Returns `(TreeVetter, (best_epoch = best round, ...))` for cv_run.
"""
function train_evotree(data; seed = 1, nrounds = 3000, eta = 0.05, max_depth = 5, min_weight = 5.0,
                       rowsample = 0.8, colsample = 0.8, L2 = 1.0, lambda = 0.0, nbins = 64,
                       early_stopping_rounds = 100, pos_weight = 1.0, verbose = false)
    config = EvoTrees.EvoTreeRegressor(; loss = :logloss, metric = :logloss, nrounds, eta, max_depth,
                                       min_weight = Float64(min_weight), rowsample, colsample,
                                       L2 = Float64(L2), lambda = Float64(lambda), nbins,
                                       early_stopping_rounds, seed)
    w = Float64.(ifelse.(data.ytr .== 1, pos_weight, 1.0))
    m = EvoTrees.fit(config; x_train = permutedims(Float64.(data.Xtr)), y_train = Float64.(data.ytr),
                     w_train = w, x_eval = permutedims(Float64.(data.Xva)), y_eval = Float64.(data.yva),
                     feature_names = string.(data.features), verbosity = verbose ? 1 : 0)
    best = m.info[:logger][:best_iter]
    return TreeVetter(m, best + 1), (best_epoch = best, val_logloss = m.info[:logger][:best_metric])
end

"Gain-based feature importance of a fitted tree model, as a DataFrame."
tree_importance(t::TreeVetter) =
    DataFrame([(feature = String(f), importance = v) for (f, v) in EvoTrees.importance(t.model)])

"Logistic-regression coefficients (on z-scored inputs) as a DataFrame, largest |coef| first."
function logreg_coefs(model, features)
    W = model[1].weight
    df = DataFrame(feature = string.(features), coef = vec(Float64.(W)))
    return sort!(df, :coef; by = abs, rev = true)
end
