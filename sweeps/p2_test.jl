# Phase 2, M6: ONE-TIME sealed-test scoring of the frozen Phase 2a models (models/p2_frozen.jld2), per D5: the same
# test split as Phase 1 (split_by_star seed 42), opened once with the user's go-ahead (10/30/2026).
#   Part 1 scores the test KOIs and saves sweeps/results/p2_test_predictions.csv. Nothing is fitted or tuned. It
#          refuses to run again if that file exists, so the test set is scored only once.
#   Part 2 evaluates the saved predictions (re-runnable: it only reads the CSV) next to Phase 1's test predictions
#          (test_predictions.csv, same KOIs) and koi_score. Writes p2_test_auc.csv and p2_test_operating_points.csv.
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p2_test.jl"
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
import Random

out = joinpath(CV_DIR, "p2_test_predictions.csv")
df_raw = load_koi()
pool, test0 = cv_pool(clean_koi(df_raw))
if !isfile(out)
    @assert isempty(intersect(Set(pool.kepid), Set(test0.kepid)))     # star-disjoint from everything trained on
    test = add_centroid_features(test0, df_raw)
    lc = CSV.read(joinpath(CV_DIR, "p2_lcfeatures.csv"), DataFrame)
    @assert all(in(Set(lc.kepoi_name)), test.kepoi_name)
    test = leftjoin(test, lc; on = :kepoi_name, order = :left)
    @assert test.kepoi_name == test0.kepoi_name
    for c in [LC_FEATURES; LC_CENTROID_FEATURES]
        test[!, c] = Float64.(coalesce.(test[!, c], NaN))
    end
    frozen = JLD2.load(joinpath(PROJECT_DIR, "models", "p2_frozen.jld2"))["frozen"]
    pred = DataFrame(kepoi_name = test.kepoi_name, kepid = test.kepid, koi_disposition = test.koi_disposition,
                     label = test.label, dr25 = is_dr25(test, df_raw),
                     p_catalog = frozen_predict(frozen["catalog_only"], test),
                     p_centroid = frozen_predict(frozen["catalog_centroid"], test))
    CSV.write(out, pred)
    @printf("scored %d test KOIs (%d stars; %d DR25) -> %s\n", nrow(pred), length(unique(pred.kepid)), sum(pred.dr25), out)
else
    println("test predictions already exist; evaluating the saved file only (no re-scoring)")
end

# ---- Part 2: evaluation of the saved predictions ----
p2 = CSV.read(out, DataFrame)
p1 = CSV.read(joinpath(CV_DIR, "test_predictions.csv"), DataFrame)
@assert p1.kepoi_name == p2.kepoi_name
frozen = JLD2.load(joinpath(PROJECT_DIR, "models", "p2_frozen.jld2"))["frozen"]
f1 = JLD2.load(joinpath(PROJECT_DIR, "models", "step5_frozen.jld2"))["frozen"]
y, drt, kep = Float64.(p2.label), Bool.(p2.dr25), p2.kepid
conft = p2.koi_disposition .== "CONFIRMED"

"Paired star bootstrap of AUC(a) - AUC(b) on the same KOIs: (delta, lo, hi)."
function paired_delta(y, a, b, kepid; B = 2000, seed = 1)
    rng = Random.Xoshiro(seed)
    stars = unique(kepid)
    rows = Dict(s => findall(==(s), kepid) for s in stars)
    d = Float64[]
    for _ in 1:B
        idx = reduce(vcat, (rows[s] for s in rand(rng, stars, length(stars))))
        length(unique(y[idx])) == 2 && push!(d, fast_auc(y[idx], a[idx]) - fast_auc(y[idx], b[idx]))
    end
    return (delta = fast_auc(y, a) - fast_auc(y, b), lo = quantile(d, 0.025), hi = quantile(d, 0.975))
end

# CV expectation (Phase 2a): one 10-seed ensemble per fold shuffle, AUC averaged over shuffles (as in step 5)
dr_pool = is_dr25(pool, df_raw); yp = Float64.(pool.label); confp = pool.koi_disposition .== "CONFIRMED"
cvS(a, b) = (repeat_scores(load_cv(a)) .+ repeat_scores(load_cv(b))) ./ 2
models = [("catalog-only", "catalog_only", p2.p_catalog, p1.p_catalog, cvS("cv_mlpq_lc_r5s10.jld2", "cv_evotree_lc_r5s10.jld2")),
          ("catalog + centroid", "catalog_centroid", p2.p_centroid, p1.p_centroid,
           cvS("cv_mlpq_lccent_r5s10.jld2", "cv_evotree_lccent_r5s10.jld2"))]
rows = NamedTuple[]
for (mname, _, p, q, S) in models
    for (sname, mt, mc) in (("DR25 KOIs (headline)", drt, dr_pool),
                            ("DR25 CONFIRMED vs FP", drt .& (conft .| (y .== 0)), dr_pool .& (confp .| (yp .== 0))),
                            ("supplemental KOIs", .!drt, .!dr_pool), ("all KOIs", trues(length(y)), trues(length(yp))))
        bs = star_bootstrap(y[mt], p[mt], kep[mt]; B = 2000)
        d = paired_delta(y[mt], p[mt], q[mt], kep[mt])
        push!(rows, (model = mname, subset = sname, n = sum(mt), test_auc = round(fast_auc(y[mt], p[mt]); digits = 4),
                     test_ci = @sprintf("[%.3f, %.3f]", bs.lo, bs.hi),
                     cv_expected = round(mean(fast_auc(yp[mc], S[mc, r]) for r in axes(S, 2)); digits = 4),
                     phase1_test_auc = round(fast_auc(y[mt], q[mt]); digits = 4),
                     delta_vs_phase1 = @sprintf("%+.4f [%+.4f, %+.4f]", d.delta, d.lo, d.hi)))
    end
end
mk = drt .& isfinite.(p1.koi_score)
push!(rows, (model = "robovetter koi_score (reference)", subset = "DR25 KOIs (headline)", n = sum(mk),
             test_auc = round(fast_auc(y[mk], p1.koi_score[mk]); digits = 4), test_ci = "—", cv_expected = NaN,
             phase1_test_auc = NaN, delta_vs_phase1 = "—"))
auc = DataFrame(rows)
CSV.write(joinpath(CV_DIR, "p2_test_auc.csv"), auc)
show(stdout, MIME"text/plain"(), auc; allcols = true); println()

# operating points at the frozen thresholds, DR25 KOIs; Phase 1's frozen model at its own thresholds alongside
cvmenu = CSV.read(joinpath(CV_DIR, "p2_m5_operating_points.csv"), DataFrame)
RULES = [("planet recall 95%", (:recall, 0.95)), ("planet recall 98%", (:recall, 0.98)),
         ("planet recall 99%", (:recall, 0.99)), ("precision 95%", (:precision, 0.95)),
         ("precision 99%", (:precision, 0.99)), ("max F1", (:maxf1,)), ("p = 0.5", (:prob, 0.5))]
ops = NamedTuple[]
for ((mname, key, p, q, _), fam) in zip(models, ("catalog", "+ centroid")), (rname, rule) in RULES
    t = frozen[key]["thresholds"][string(rule)]
    t1 = f1[key]["thresholds"][string(rule)]
    o, o1 = op_metrics(y[drt], p[drt], t), op_metrics(y[drt], q[drt], t1)
    rb = star_bootstrap(y[drt], p[drt], kep[drt]; B = 1000, metric = (a, b) -> op_metrics(a, b, t).recall)
    pb = star_bootstrap(y[drt], p[drt], kep[drt]; B = 1000, metric = (a, b) -> op_metrics(a, b, t).precision)
    mc = drt .& (conft .| (y .== 0))
    cvr = cvmenu[(cvmenu.family .== fam) .& (cvmenu.phase .== "Phase 2a") .& (cvmenu.rule .== rname), :]
    push!(ops, (model = mname, rule = rname, t = round(t; digits = 3),
                recall = @sprintf("%.3f [%.3f, %.3f]", o.recall, rb.lo, rb.hi), cv_recall = cvr.dr25_recall[1],
                precision = @sprintf("%.3f [%.3f, %.3f]", o.precision, pb.lo, pb.hi), cv_precision = cvr.dr25_precision[1],
                fp_kept = round(o.fp_kept; digits = 3), cv_fp_kept = cvr.dr25_fp_kept[1],
                confirmed_lost = round(1 - op_metrics(y[mc], p[mc], t).recall; digits = 3),
                phase1_recall = round(o1.recall; digits = 3), phase1_precision = round(o1.precision; digits = 3),
                phase1_fp_kept = round(o1.fp_kept; digits = 3)))
end
ops = DataFrame(ops)
CSV.write(joinpath(CV_DIR, "p2_test_operating_points.csv"), ops)
show(stdout, MIME"text/plain"(), ops; allcols = true, allrows = true); println()
println("DONE")
