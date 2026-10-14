# Step 5: freeze the two final models before the one-time sealed-test evaluation.
#   catalog-only     : MLP (lr 3e-3, 600 epochs, patience 50, :quantile) + tuned trees, 50/50 average
#   catalog+centroid : the same, with FEATURES + CENTROID_FEATURES
# Each = 10 seeds of each learner, trained on the WHOLE pool (inner 1/9 of stars held out for early stopping,
# exactly as inside cv_run). Also stores the operating-point menu (thresholds chosen on the pool's DR25 OOF
# scores, exo101326 section 11). The test split is NOT touched here.
#   julia -t auto sweeps/step5_freeze.jl
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
using Dates

df_raw = load_koi()
df = clean_koi(df_raw)
pool0, _ = cv_pool(df)
pool = add_centroid_features(pool0, df_raw)
y = Float64.(pool.label)
dr = is_dr25(pool, df_raw)
const MLP_KW  = (lr = 3e-3, epochs = 600, patience = 50)
const TREE_KW = (max_depth = 8, min_weight = 10.0, L2 = 10.0, colsample = 0.8)
const SEEDS = 1:10
const RULES = [(:recall, 0.95), (:recall, 0.98), (:recall, 0.99), (:precision, 0.95), (:precision, 0.99),
               (:maxf1,), (:prob, 0.5)]
specs = [("catalog_only", FEATURES, "cv_mlp_lr3_e600_quantile_r5s10.jld2", "cv_evotree_tuned_r5s10.jld2"),
         ("catalog_centroid", [FEATURES; CENTROID_FEATURES], "cv_mlpq_centroid_r5s10.jld2", "cv_evotree_centroid_r5s10.jld2")]

tr, va = split_by_star(pool; at = (8 / 9,), seed = 5005)
frozen = Dict{String, Any}()
for (name, feats, mlp_file, tree_file) in specs
    t0 = time()
    data = prepare_data(tr, va, va; features = feats, transform = :quantile)
    mlps = [Flux.state(first(train_model(data; seed = s, verbose = false, MLP_KW...))) for s in SEEDS]
    trees = [first(train_evotree(data; seed = s, TREE_KW...)) for s in SEEDS]
    # operating-point menu: per fold shuffle, threshold on the DR25 KOIs' OOF scores; frozen value = median
    S = (repeat_scores(load_cv(mlp_file)) .+ repeat_scores(load_cv(tree_file))) ./ 2
    thr = Dict(string(r) => median([pick_threshold(y[dr], S[dr, k], r) for k in axes(S, 2)]) for r in RULES)
    frozen[name] = Dict("features" => string.(feats), "prep" => (features = feats, transform = data.transform,
                        scaler = data.scaler, qref = data.qref, med = data.med),
                        "mlp_states" => mlps, "trees" => trees, "thresholds" => thr, "calibrator" => "none")
    @printf("%s: frozen in %.1f min; thresholds %s\n", name, (time() - t0) / 60, thr)
end
frozen["info"] = Dict("pool_kois" => nrow(pool), "inner_split_seed" => 5005, "seeds" => collect(SEEDS),
                      "mlp_kw" => Dict(string(k) => v for (k, v) in pairs(MLP_KW)),
                      "tree_kw" => Dict(string(k) => v for (k, v) in pairs(TREE_KW)),
                      "hidden" => (64, 32), "dropout" => 0.2, "created" => string(Dates.now()))
path = joinpath(PROJECT_DIR, "models", "step5_frozen.jld2")
jldsave(path; frozen)
println("saved: ", path)

# sanity check on the pool itself (in-sample, so optimistic; only checks the round trip works)
reload = JLD2.load(path)["frozen"]
for (name, _) in specs
    p = frozen_predict(reload[name], pool)
    @printf("%s: in-sample pool AUC %.4f (DR25 %.4f), mean p %.3f\n", name, fast_auc(y, p), fast_auc(y[dr], p[dr]), mean(p))
end
println("DONE")
