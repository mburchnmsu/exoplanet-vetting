# ==============================================================================
# CROSS-VALIDATION: star-grouped k-fold CV, repeated over fold shuffles and model
# seeds, with out-of-fold (OOF) predictions for every KOI in the pool.
# Included automatically by VettingSetup.jl.
#
# The test split from split_by_star(df; seed = 42) is never passed in here:
# CV runs on the train+val "pool" only.
# ==============================================================================

"""
    star_folds(df; k = 5, seed = 1) -> Vector{Int}

Fold index (1..k) for every row of `df`. All KOIs of a star share a fold, and
stars are dealt round-robin within each system type (planet / fp / mixed) so
every fold has about the same label balance.
"""
function star_folds(df::DataFrame; k = 5, seed = 1)
    rng = Xoshiro(seed)
    stars = star_systems(df)
    fold_of = Dict{eltype(stars.kepid), Int}()
    for g in groupby(stars, :system)
        offset = rand(rng, 0:k-1)
        for (j, i) in enumerate(shuffle(rng, 1:nrow(g)))
            fold_of[g.kepid[i]] = mod1(j + offset, k)
        end
    end
    return [fold_of[id] for id in df.kepid]
end

"Pool = train + val of the main split (test excluded)."
function cv_pool(df::DataFrame; split_seed = 42)
    train, val, test = split_by_star(df; seed = split_seed)
    return vcat(train, val), test
end

"ROC-AUC via the Mann-Whitney rank statistic (ties get average ranks). Fast; used for bootstraps."
function fast_auc(y::AbstractVector, p::AbstractVector)
    r = tiedrank(p)
    pos = y .== 1
    n1, n0 = sum(pos), length(y) - sum(pos)
    return (sum(r[pos]) - n1 * (n1 + 1) / 2) / (n1 * n0)
end

"Mean binary cross-entropy of probabilities (unweighted), clipped for safety."
logloss(y, p) = -mean(@. y * log(clamp(p, 1e-7, 1 - 1e-7)) + (1 - y) * log(clamp(1 - p, 1e-7, 1 - 1e-7)))

"Metrics on one held-out set, including the CONFIRMED-vs-FP subset."
function fold_metrics(df::DataFrame, p; threshold = 0.5)
    m = evaluate(df.label, p; threshold)
    c = df.koi_disposition .!= "CANDIDATE"
    return (roc_auc = m.roc_auc, pr_auc = m.pr_auc, accuracy = m.accuracy, f1 = m.f1,
            precision = m.precision, recall = m.recall, logloss = logloss(df.label, p),
            roc_auc_conf = au_roccurve(Int.(df.label[c]), p[c]),
            pr_auc_conf = au_prcurve(Int.(df.label[c]), p[c]))
end

"""
    cv_run(pool; k = 5, repeats = 3, seeds = 1:10, features = FEATURES,
           inner_val = 1/9, fold_seed0 = 1000, name = "cv", train_kw...)

For each repeat r (a fresh fold shuffle) and fold f: train on the other k-1
folds, holding out `inner_val` of their stars for early stopping (the scaler is
fit on the inner training part only), then predict fold f. Repeated for every
model seed. Returns a NamedTuple:

- `results`: one row per (repeat, fold, seed) with held-out fold metrics
- `oof`:     [n_pool, repeats, n_seeds] out-of-fold planet probabilities
- `folds`:   [n_pool, repeats] fold assignment
- `config`:  settings used

`trainer(data; seed, verbose, train_kw...)` must return `(model, hist)` with
`hist.best_epoch`, and `predict_proba(model, X)` must work on the result.
Default is the MLP's `train_model`; Baselines.jl adds `train_evotree`.
"""
function cv_run(pool::DataFrame; k = 5, repeats = 3, seeds = 1:10, features = FEATURES,
                inner_val = 1 / 9, fold_seed0 = 1000, name = "cv", verbose = true,
                trainer = train_model, transform = :default, train_filter = nothing, extra = nothing, train_kw...)
    seeds = collect(seeds)
    n = nrow(pool)
    oof = fill(NaN, n, repeats, length(seeds))
    folds = zeros(Int, n, repeats)
    rows = NamedTuple[]
    t0 = time()
    for r in 1:repeats
        folds[:, r] = star_folds(pool; k, seed = fold_seed0 + r)
        for f in 1:k
            held = folds[:, r] .== f
            # train_filter (Bool per pool row) restricts which KOIs are TRAINED on; folds and
            # predictions still cover every pool KOI, so runs stay paired with the baseline
            fit_rows = train_filter === nothing ? .!held : (.!held .& train_filter)
            fit_df, test_df = pool[fit_rows, :], pool[held, :]
            tr, va = split_by_star(fit_df; at = (1 - inner_val,), seed = fold_seed0 + 100r + f)
            data = prepare_data(tr, va, test_df; features, transform)
            # extra(tr, va, te) adds per-KOI arrays (e.g. light-curve views for the CNN) to the fold's data
            extra === nothing || (data = merge(data, extra(tr, va, test_df)))
            for (si, s) in enumerate(seeds)
                model, hist = trainer(data; seed = s, verbose = false, train_kw...)
                p = Float64.(predict_test(model, data))
                oof[held, r, si] = p
                push!(rows, merge((name = name, repeat = r, fold = f, seed = s,
                                   best_epoch = hist.best_epoch, n_test = nrow(test_df)),
                                  fold_metrics(test_df, p)))
            end
            verbose && @printf("[%s] repeat %d fold %d done  (%.1f min elapsed)\n", name, r, f, (time() - t0) / 60)
        end
    end
    config = Dict("k" => k, "repeats" => repeats, "seeds" => seeds, "features" => string.(features),
                  "trainer" => string(nameof(trainer)), "transform" => string(transform),
                  "train_filter" => train_filter === nothing ? "none" : "$(sum(train_filter)) of $(length(train_filter)) KOIs",
                  "inner_val" => inner_val, "fold_seed0" => fold_seed0,
                  "train_kw" => Dict(string(kk) => v for (kk, v) in pairs(train_kw)))
    return (results = DataFrame(rows), oof = oof, folds = folds, config = config,
            kepoi = String.(pool.kepoi_name))
end

"Held-out predictions of a trained model on a fold's test part (models needing more than `Xte` add a method)."
predict_test(model, data) = predict_proba(model, data.Xte)

const CV_DIR = joinpath(PROJECT_DIR, "sweeps", "results")

function save_cv(file, cv)
    mkpath(CV_DIR)
    path = joinpath(CV_DIR, file)
    jldsave(path; results = cv.results, oof = cv.oof, folds = cv.folds, config = cv.config, kepoi = cv.kepoi)
    return path
end

function load_cv(file)
    d = JLD2.load(joinpath(CV_DIR, file))
    return (results = d["results"], oof = d["oof"], folds = d["folds"], config = d["config"], kepoi = d["kepoi"])
end

# --- Summaries -----------------------------------------------------------------

"""
    oof_scores(cv; seeds = :) -> Vector

Final out-of-fold score per pool KOI: mean over the chosen seeds and over repeats
(each repeat's prediction for a KOI comes from a model that never saw its star).
"""
oof_scores(cv; seeds = Colon()) = vec(mean(cv.oof[:, :, seeds]; dims = (2, 3)))

"""
    noise_summary(cv, pool; metric = fast_auc)

Breaks the spread of a metric into its sources:
- single model on one fold (what one notebook run with a ~930-KOI val set gives you)
- seed-to-seed spread with the fold fixed
- fold-to-fold spread of the seed-ensembled score
- pooled OOF score per repeat (single seed vs seed ensemble)
- null difference: two disjoint seed groups on identical folds = two runs of the
  *same* config, so any gap between them is noise
"""
function noise_summary(cv, pool::DataFrame; metric = fast_auc)
    y = pool.label
    R, S = size(cv.oof, 2), size(cv.oof, 3)
    k = maximum(cv.folds)
    single = [metric(y[cv.folds[:, r] .== f], cv.oof[cv.folds[:, r] .== f, r, s]) for r in 1:R, f in 1:k, s in 1:S]
    ens = [metric(y[cv.folds[:, r] .== f], vec(mean(cv.oof[cv.folds[:, r] .== f, r, :]; dims = 2))) for r in 1:R, f in 1:k]
    pooled_single = [metric(y, cv.oof[:, r, s]) for r in 1:R, s in 1:S]
    pooled_ens = [metric(y, vec(mean(cv.oof[:, r, :]; dims = 2))) for r in 1:R]
    h = S ÷ 2
    gA, gB = 1:h, h+1:2h
    null_fold = [metric(y[cv.folds[:, r] .== f], vec(mean(cv.oof[cv.folds[:, r] .== f, r, gA]; dims = 2))) -
                 metric(y[cv.folds[:, r] .== f], vec(mean(cv.oof[cv.folds[:, r] .== f, r, gB]; dims = 2))) for r in 1:R, f in 1:k]
    null_pooled = metric(y, oof_scores(cv; seeds = gA)) - metric(y, oof_scores(cv; seeds = gB))
    return (single = single, ens = ens, pooled_single = pooled_single, pooled_ens = pooled_ens,
            null_fold = null_fold, null_pooled = null_pooled,
            final = metric(y, oof_scores(cv)),
            seed_std_within_fold = mean(std(single; dims = 3)),
            fold_std_single = std(single), fold_std_ens = std(ens),
            ensemble_gain = mean(ens) - mean(single))
end

"""
    null_differences(cv, pool; n = 500, metric = fast_auc, seed = 11)

Distribution of the pooled-OOF metric difference between two random disjoint
halves of the seeds, on identical folds. Both halves are the *same* config, so
this is the spread you would see between two runs that differ only by chance.
A real improvement should clear it comfortably.
"""
function null_differences(cv, pool::DataFrame; n = 500, metric = fast_auc, seed = 11)
    rng = Xoshiro(seed)
    S = size(cv.oof, 3); h = S ÷ 2
    return [begin
                perm = randperm(rng, S)
                metric(pool.label, oof_scores(cv; seeds = perm[1:h])) - metric(pool.label, oof_scores(cv; seeds = perm[h+1:2h]))
            end for _ in 1:n]
end

"""
    compare_cv(cvA, cvB, pool; seeds = :, B = 2000, metric = fast_auc, null = nothing)

Compare two configs run with `cv_run` on the SAME pool and fold seeds (paired).
Returns ΔAUC = A - B on pooled OOF scores, a paired star-bootstrap 95% CI on Δ
(does the gain hold up for other samples of stars?), and, if a null distribution
from `null_differences` is supplied, the fraction of null |Δ| at least as large
(is the gain bigger than training noise?).
Both tests should pass before calling a change an improvement.
"""
function compare_cv(cvA, cvB, pool::DataFrame; seeds = Colon(), B = 2000, metric = fast_auc, null = nothing, seed = 5)
    @assert cvA.kepoi == cvB.kepoi == String.(pool.kepoi_name) "runs must use the same pool, in the same order"
    @assert cvA.folds == cvB.folds "runs must use the same folds (same k, repeats, fold_seed0)"
    y = pool.label
    sA, sB = oof_scores(cvA; seeds), oof_scores(cvB; seeds)
    Δ = metric(y, sA) - metric(y, sB)
    rng = Xoshiro(seed)
    ids = unique(pool.kepid)
    rows_of = Dict(id => Int[] for id in ids)
    for (i, id) in enumerate(pool.kepid); push!(rows_of[id], i); end
    boot = map(1:B) do _
        idx = reduce(vcat, [rows_of[id] for id in rand(rng, ids, length(ids))])   # a Vector, not a generator: one allocation instead of O(n²) copying
        metric(y[idx], sA[idx]) - metric(y[idx], sB[idx])
    end
    p_null = null === nothing ? missing : mean(abs.(null) .>= abs(Δ))
    return (delta = Δ, A = metric(y, sA), B = metric(y, sB),
            ci_lo = quantile(boot, 0.025), ci_hi = quantile(boot, 0.975), p_null = p_null)
end

"""
    star_bootstrap(y, p, kepid; B = 2000, metric = fast_auc, seed = 7)

Bootstrap over stars (resample stars with replacement, keep all their KOIs) to
put a confidence interval on a pooled metric.
"""
function star_bootstrap(y, p, kepid; B = 2000, metric = fast_auc, seed = 7)
    rng = Xoshiro(seed)
    ids = unique(kepid)
    rows_of = Dict(id => Int[] for id in ids)
    for (i, id) in enumerate(kepid); push!(rows_of[id], i); end
    vals = Vector{Float64}(undef, B)
    for b in 1:B
        idx = reduce(vcat, [rows_of[id] for id in rand(rng, ids, length(ids))])
        vals[b] = metric(y[idx], p[idx])
    end
    return (mean = mean(vals), lo = quantile(vals, 0.025), hi = quantile(vals, 0.975), samples = vals)
end
