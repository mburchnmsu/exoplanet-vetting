# Phase 2, M5: the Phase 2a models (MLP + trees 50/50, with the light-curve features) become the Phase 2 models
# (M4: the CNNs neither beat nor add to them). Same procedure as Phase 1 step 5 (src/Calibration.jl).
#   --analyze : from the saved CV only (light): nested calibration check, operating-point menu, and the plan's §1
#               success criteria, Phase 1 vs Phase 2a on identical folds. Writes sweeps/results/p2_m5_*.csv.
#   --freeze  : train both Phase 2a models on the whole pool (10 seeds each learner, fixed inner split as in
#               step5_freeze.jl) + menu thresholds -> models/p2_frozen.jld2. The test split is NOT touched.
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p2_m5.jl --analyze"
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
using Dates

df_raw = load_koi()
pool0, _ = cv_pool(clean_koi(df_raw))
pool = add_centroid_features(pool0, df_raw)
pool = leftjoin(pool, CSV.read(joinpath(CV_DIR, "p2_lcfeatures.csv"), DataFrame); on = :kepoi_name, order = :left)
@assert pool.kepoi_name == pool0.kepoi_name
for c in [LC_FEATURES; LC_CENTROID_FEATURES]
    pool[!, c] = Float64.(coalesce.(pool[!, c], NaN))
end
y = Float64.(pool.label)
dr = is_dr25(pool, df_raw)
conf = pool.koi_disposition .== "CONFIRMED"
const LCF  = [FEATURES; LC_FEATURES]
const LCCF = [FEATURES; CENTROID_FEATURES; LC_FEATURES; LC_CENTROID_FEATURES]
const RULES = [("planet recall 95%", (:recall, 0.95)), ("planet recall 98%", (:recall, 0.98)),
               ("planet recall 99%", (:recall, 0.99)), ("precision 95%", (:precision, 0.95)),
               ("precision 99%", (:precision, 0.99)), ("max F1", (:maxf1,)), ("p = 0.5", (:prob, 0.5))]
# (family, key, features, phase, MLP CV, trees CV)
const MODELS = [("catalog", "catalog_only", FEATURES, "Phase 1", "cv_mlp_lr3_e600_quantile_r5s10.jld2", "cv_evotree_tuned_r5s10.jld2"),
                ("catalog", "catalog_only", LCF, "Phase 2a", "cv_mlpq_lc_r5s10.jld2", "cv_evotree_lc_r5s10.jld2"),
                ("+ centroid", "catalog_centroid", [FEATURES; CENTROID_FEATURES], "Phase 1", "cv_mlpq_centroid_r5s10.jld2", "cv_evotree_centroid_r5s10.jld2"),
                ("+ centroid", "catalog_centroid", LCCF, "Phase 2a", "cv_mlpq_lccent_r5s10.jld2", "cv_evotree_lccent_r5s10.jld2")]
"[n_pool, repeats] seed-mean OOF score per fold shuffle of the 50/50 MLP + trees average, and the folds."
function scores(mlp_file, tree_file)
    a, b = load_cv(mlp_file), load_cv(tree_file)
    @assert a.kepoi == pool.kepoi_name && a.folds == b.folds
    return (repeat_scores(a) .+ repeat_scores(b)) ./ 2, a.folds
end
r3(x) = round(x; digits = 3)
r4(x) = round(x; digits = 4)
show_df(t) = (show(stdout, MIME"text/plain"(), t; allcols = true, allrows = true); println(); flush(stdout))

if "--analyze" in ARGS
    # 1. calibration (Phase 2a only; Phase 1's is in step 5): nested none / Platt / isotonic
    cal = NamedTuple[]
    for (fam, _, _, phase, fm, ft) in MODELS
        phase == "Phase 2a" || continue
        S, folds = scores(fm, ft)
        c = fit_platt(vec(S), repeat(y, size(S, 2)))
        @printf("%s %s: Platt fit on all OOF scores a = %.3f, b = %+.3f (1, 0 = already calibrated)\n", fam, phase, c.a, c.b)
        for meth in (:none, :platt, :isotonic)
            P = nested_calibrate(S, folds, y, meth)
            for (sname, m) in (("DR25", dr), ("supplemental", .!dr), ("all", trues(length(y))))
                mr(f) = mean(f(y[m], P[m, r]) for r in axes(P, 2))
                push!(cal, (family = fam, calibrator = string(meth), subset = sname, logloss = r4(mr(logloss)),
                            brier = r4(mr(brier)), ece = r4(mr(ece)), roc_auc = r4(mr(fast_auc)),
                            mean_p = r3(mean(P[m, :])), planet_rate = r3(mean(y[m]))))
            end
        end
    end
    cal = DataFrame(cal); CSV.write(joinpath(CV_DIR, "p2_m5_calibration.csv"), cal); show_df(cal)

    # 2. operating-point menu: nested thresholds chosen on DR25 KOIs, one threshold for every KOI
    menu = NamedTuple[]
    calls98 = Dict{Tuple{String, String}, BitMatrix}()
    calls05 = Dict{Tuple{String, String}, BitMatrix}()
    for (fam, _, _, phase, fm, ft) in MODELS
        S, folds = scores(fm, ft)
        for (rname, rule) in RULES
            nc = nested_calls(S, folds, y, dr, rule)
            rname == "planet recall 98%" && (calls98[(fam, phase)] = nc.calls)
            rname == "p = 0.5" && (calls05[(fam, phase)] = nc.calls)
            rate(m, f) = mean(f(op_metrics(y[m], Float64.(nc.calls[m, r]), 0.5)) for r in axes(S, 2))
            push!(menu, (family = fam, phase = phase, rule = rname,
                         t_median = r3(median(filter(isfinite, vec(nc.t)))), t_spread = r3(std(nc.t)),
                         dr25_recall = r3(rate(dr, o -> o.recall)), dr25_precision = r3(rate(dr, o -> o.precision)),
                         dr25_fp_kept = r3(rate(dr, o -> o.fp_kept)),
                         planets_lost_per1000 = round(Int, 1000 * rate(dr, o -> o.fn) / sum(dr)),
                         fps_kept_per1000 = round(Int, 1000 * rate(dr, o -> o.fp) / sum(dr)),
                         confirmed_lost = r3(rate(dr .& (conf .| (y .== 0)), o -> 1 - o.recall)),
                         supp_recall = r3(rate(.!dr, o -> o.recall)), supp_precision = r3(rate(.!dr, o -> o.precision))))
        end
    end
    menu = DataFrame(menu); CSV.write(joinpath(CV_DIR, "p2_m5_operating_points.csv"), menu); show_df(menu)

    # 3. §1 criteria: blind-spot groups (nested hard calls, mean over fold shuffles) at p = 0.5 and at 98% recall
    a = with_vetting_info(pool, df_raw, zeros(nrow(pool)))
    fp = y .== 0
    groups = [("FP missed: NT only", fp .& (a.flag_pattern .== "NT"), true),
              ("FP missed: EC only", fp .& (a.flag_pattern .== "EC"), true),
              ("FP missed: CO only", fp .& (a.flag_pattern .== "CO"), true),
              ("FP missed: SS only", fp .& (a.flag_pattern .== "SS"), true),
              ("FP missed: all", fp, true),
              ("CONFIRMED called FP", conf, false),
              ("CONFIRMED called FP (DR25)", conf .& dr, false),
              ("planet called FP: P < 1.5 d", (y .== 1) .& (pool.koi_period .< 1.5), false),
              ("planet called FP: all", y .== 1, false)]
    crit = NamedTuple[]
    for (gname, m, isfp) in groups, (fam, _, _, phase, _, _) in MODELS
        err(C) = mean(mean(isfp ? C[m, r] : .!C[m, r]) for r in axes(C, 2))
        push!(crit, (group = gname, n = sum(m), family = fam, phase = phase,
                     pct_at_p05 = round(100 * err(calls05[(fam, phase)]); digits = 1),
                     pct_at_recall98 = round(100 * err(calls98[(fam, phase)]); digits = 1)))
    end
    crit = DataFrame(crit); CSV.write(joinpath(CV_DIR, "p2_m5_criteria.csv"), crit); show_df(crit)
    for (fam, _, _, phase, fm, ft) in MODELS
        S, _ = scores(fm, ft)
        @printf("%-10s %-8s supplemental AUC (mean over fold shuffles) %.4f; DR25 %.4f\n", fam, phase,
                mean(fast_auc(y[.!dr], S[.!dr, r]) for r in axes(S, 2)), mean(fast_auc(y[dr], S[dr, r]) for r in axes(S, 2)))
    end
end

if "--freeze" in ARGS
    MLP_KW  = (lr = 3e-3, epochs = 600, patience = 50)
    TREE_KW = (max_depth = 8, min_weight = 10.0, L2 = 10.0, colsample = 0.8)
    SEEDS = 1:10
    tr, va = split_by_star(pool; at = (8 / 9,), seed = 5005)      # the same inner split as Phase 1's freeze
    frozen = Dict{String, Any}()
    for (fam, key, feats, phase, fm, ft) in MODELS
        phase == "Phase 2a" || continue
        t0 = time()
        data = prepare_data(tr, va, va; features = feats, transform = :quantile)
        mlps = [Flux.state(first(train_model(data; seed = s, verbose = false, MLP_KW...))) for s in SEEDS]
        trees = [first(train_evotree(data; seed = s, TREE_KW...)) for s in SEEDS]
        S, _ = scores(fm, ft)
        thr = Dict(string(rule) => median([pick_threshold(y[dr], S[dr, k], rule) for k in axes(S, 2)]) for (_, rule) in RULES)
        frozen[key] = Dict("features" => string.(feats), "prep" => (features = feats, transform = data.transform,
                           scaler = data.scaler, qref = data.qref, med = data.med),
                           "mlp_states" => mlps, "trees" => trees, "thresholds" => thr, "calibrator" => "none")
        @printf("%s: frozen in %.1f min; thresholds %s\n", key, (time() - t0) / 60, thr); flush(stdout)
    end
    frozen["info"] = Dict("pool_kois" => nrow(pool), "inner_split_seed" => 5005, "seeds" => collect(SEEDS),
                          "mlp_kw" => Dict(string(k) => v for (k, v) in pairs(MLP_KW)),
                          "tree_kw" => Dict(string(k) => v for (k, v) in pairs(TREE_KW)),
                          "hidden" => (64, 32), "dropout" => 0.2, "lc_features" => "sweeps/results/p2_lcfeatures.csv",
                          "created" => string(Dates.now()))
    path = joinpath(PROJECT_DIR, "models", "p2_frozen.jld2")
    jldsave(path; frozen)
    println("saved: ", path)
    reload = JLD2.load(path)["frozen"]            # round-trip check on the pool (in-sample, so optimistic)
    for key in ("catalog_only", "catalog_centroid")
        p = frozen_predict(reload[key], pool)
        @printf("%s: in-sample pool AUC %.4f (DR25 %.4f), mean p %.3f\n", key, fast_auc(y, p), fast_auc(y[dr], p[dr]), mean(p))
    end
end
println(Dates.format(now(), "mm-dd HH:MM"), "  DONE")
