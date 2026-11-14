# Phase 3, adoption (user decision 11/13): freeze the two "+ ephemeris" models (MLP + trees 50/50; Phase 2a features
# + EPHEMERIS_FEATURES, v1 matching without the EB catalog). Same procedure as sweeps/p2_m5.jl --freeze: whole pool,
# 10 seeds per learner, the fixed inner split, menu thresholds from the saved CV (nested, picked on DR25 KOIs).
# Post-test: the sealed test split is spent and NOT touched; these models are "post-test, CV-validated".
#   -> models/p3_frozen.jld2 (keys catalog_ephemeris, centroid_ephemeris)
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p3_freeze.jl"
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
using Dates

df_raw = load_koi()
pool0, _ = cv_pool(clean_koi(df_raw))
pool = add_centroid_features(pool0, df_raw)
pool = leftjoin(pool, CSV.read(joinpath(CV_DIR, "p2_lcfeatures.csv"), DataFrame); on = :kepoi_name, order = :left)
eph = CSV.read(joinpath(CV_DIR, "p3_ephemeris.csv"), DataFrame)
pool = leftjoin(pool, eph[:, [:kepoi_name; EPHEMERIS_FEATURES]]; on = :kepoi_name, order = :left)
@assert pool.kepoi_name == pool0.kepoi_name
for c in [LC_FEATURES; LC_CENTROID_FEATURES; EPHEMERIS_FEATURES]
    pool[!, c] = Float64.(coalesce.(pool[!, c], NaN))
end
y = Float64.(pool.label)
dr = is_dr25(pool, df_raw)
const RULES = [(:recall, 0.95), (:recall, 0.98), (:recall, 0.99), (:precision, 0.95), (:precision, 0.99), (:maxf1,),
               (:prob, 0.5)]
const MODELS = [("catalog_ephemeris", [FEATURES; LC_FEATURES; EPHEMERIS_FEATURES],
                 "cv_mlpq_eph_r5s10.jld2", "cv_evotree_eph_r5s10.jld2"),
                ("centroid_ephemeris", [FEATURES; CENTROID_FEATURES; LC_FEATURES; LC_CENTROID_FEATURES; EPHEMERIS_FEATURES],
                 "cv_mlpq_centeph_r5s10.jld2", "cv_evotree_centeph_r5s10.jld2")]
MLP_KW  = (lr = 3e-3, epochs = 600, patience = 50)
TREE_KW = (max_depth = 8, min_weight = 10.0, L2 = 10.0, colsample = 0.8)
SEEDS = 1:10
tr, va = split_by_star(pool; at = (8 / 9,), seed = 5005)          # the same inner split as Phase 1/2 freezes
frozen = Dict{String, Any}()
for (key, feats, fm, ft) in MODELS
    t0 = time()
    a, b = load_cv(fm), load_cv(ft)
    @assert a.kepoi == pool.kepoi_name && a.folds == b.folds
    S = (repeat_scores(a) .+ repeat_scores(b)) ./ 2
    data = prepare_data(tr, va, va; features = feats, transform = :quantile)
    mlps = [Flux.state(first(train_model(data; seed = s, verbose = false, MLP_KW...))) for s in SEEDS]
    trees = [first(train_evotree(data; seed = s, TREE_KW...)) for s in SEEDS]
    thr = Dict(string(rule) => median([pick_threshold(y[dr], S[dr, k], rule) for k in axes(S, 2)]) for rule in RULES)
    frozen[key] = Dict("features" => string.(feats), "prep" => (features = feats, transform = data.transform,
                       scaler = data.scaler, qref = data.qref, med = data.med),
                       "mlp_states" => mlps, "trees" => trees, "thresholds" => thr, "calibrator" => "none")
    @printf("%s: frozen in %.1f min; thresholds %s\n", key, (time() - t0) / 60, thr); flush(stdout)
end
frozen["info"] = Dict("pool_kois" => nrow(pool), "inner_split_seed" => 5005, "seeds" => collect(SEEDS),
                      "mlp_kw" => Dict(string(k) => v for (k, v) in pairs(MLP_KW)),
                      "tree_kw" => Dict(string(k) => v for (k, v) in pairs(TREE_KW)),
                      "hidden" => (64, 32), "dropout" => 0.2, "lc_features" => "sweeps/results/p2_lcfeatures.csv",
                      "ephemeris_features" => "sweeps/results/p3_ephemeris.csv (v1, no EB catalog)",
                      "status" => "post-test, CV-validated (sealed test spent in Phase 2; not re-scored)",
                      "created" => string(Dates.now()))
path = joinpath(PROJECT_DIR, "models", "p3_frozen.jld2")
jldsave(path; frozen)
println("saved: ", path)
reload = JLD2.load(path)["frozen"]                # round-trip check on the pool (in-sample, so optimistic)
for (key, _, _, _) in MODELS
    p = frozen_predict(reload[key], pool)
    @printf("%s: in-sample pool AUC %.4f (DR25 %.4f), mean p %.3f\n", key, fast_auc(y, p), fast_auc(y[dr], p[dr]), mean(p))
end
println(Dates.format(now(), "mm-dd HH:MM"), "  DONE")
