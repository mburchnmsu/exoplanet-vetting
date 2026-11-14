# Phase 4, M2 part 2: analysis of the "+ NT" CV (sweeps/p4_cv.jl), from the saved CV only. Same machinery as
# sweeps/p3_m2.jl. Writes sweeps/results/p4_m2_*.csv (default: + NT vs Phase 2a) or p4_m2eph_*.csv (--eph: the
# adopted "+ ephemeris" lineage, + eph + NT vs + ephemeris):
#   summary   paired ROC-AUC (MLP + trees 50/50), DR25 / supplemental / all
#   menu      nested operating points (thresholds chosen on DR25 KOIs)
#   criteria  blind-spot groups at p = 0.5 and 98% recall, with a paired star-bootstrap CI on the change (plan N5)
#   lowfp     planet recall at 5% and 10% FP rate on DR25 KOIs vs the robovetter's koi_score (reference only)
#   julia --project=. -t 2 sweeps/p4_m2.jl [--eph]
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
using Dates
const EPH = "--eph" in ARGS
const OLD, NEW = EPH ? ("+ ephemeris", "+ eph + NT") : ("Phase 2a", "+ NT")
const TAG = EPH ? "p4_m2eph" : "p4_m2"

df_raw = load_koi()
pool, _ = cv_pool(clean_koi(df_raw))
y = Float64.(pool.label)
dr = is_dr25(pool, df_raw)
conf = pool.koi_disposition .== "CONFIRMED"
const RULES = [("planet recall 95%", (:recall, 0.95)), ("planet recall 98%", (:recall, 0.98)),
               ("planet recall 99%", (:recall, 0.99)), ("precision 95%", (:precision, 0.95)),
               ("max F1", (:maxf1,)), ("p = 0.5", (:prob, 0.5))]
# (family, model, MLP CV, trees CV)
const MODELS = EPH ?
    [("catalog", OLD, "cv_mlpq_eph_r5s10.jld2", "cv_evotree_eph_r5s10.jld2"),
     ("catalog", NEW, "cv_mlpq_ephnt_r5s10.jld2", "cv_evotree_ephnt_r5s10.jld2"),
     ("+ centroid", OLD, "cv_mlpq_centeph_r5s10.jld2", "cv_evotree_centeph_r5s10.jld2"),
     ("+ centroid", NEW, "cv_mlpq_centephnt_r5s10.jld2", "cv_evotree_centephnt_r5s10.jld2")] :
    [("catalog", OLD, "cv_mlpq_lc_r5s10.jld2", "cv_evotree_lc_r5s10.jld2"),
     ("catalog", NEW, "cv_mlpq_nt_r5s10.jld2", "cv_evotree_nt_r5s10.jld2"),
     ("+ centroid", OLD, "cv_mlpq_lccent_r5s10.jld2", "cv_evotree_lccent_r5s10.jld2"),
     ("+ centroid", NEW, "cv_mlpq_centnt_r5s10.jld2", "cv_evotree_centnt_r5s10.jld2")]
r3(x) = round(x; digits = 3)
r4(x) = round(x; digits = 4)
show_df(t) = (show(stdout, MIME"text/plain"(), t; allcols = true, allrows = true); println(); flush(stdout))
avg(a, b) = (A = load_cv(a); B = load_cv(b); @assert A.kepoi == pool.kepoi_name && A.folds == B.folds;
             merge(A, (oof = (A.oof .+ B.oof) ./ 2,)))
sub(cv, m) = merge(cv, (oof = cv.oof[m, :, :], folds = cv.folds[m, :], kepoi = cv.kepoi[m]))
cvs = Dict((f, p) => avg(fm, ft) for (f, p, fm, ft) in MODELS)
S = Dict(k => repeat_scores(v) for (k, v) in cvs)          # [n_pool, repeats] seed-mean OOF scores
folds = cvs[("catalog", OLD)].folds

# 1. paired AUC, NEW vs OLD
rows = NamedTuple[]
for fam in ("catalog", "+ centroid"), (lab, m) in [("DR25 KOIs", dr), ("supplemental KOIs", .!dr), ("all pool", trues(length(y)))]
    c = compare_cv(sub(cvs[(fam, NEW)], m), sub(cvs[(fam, OLD)], m), pool[m, :]; B = 1000)
    push!(rows, (family = fam, subset = lab, n = sum(m), old = OLD, auc_old = r4(c.B), new = NEW, auc_new = r4(c.A),
                 delta = r4(c.delta), ci_lo = r4(c.ci_lo), ci_hi = r4(c.ci_hi)))
end
summ = DataFrame(rows); CSV.write(joinpath(CV_DIR, "$(TAG)_summary.csv"), summ); show_df(summ)

# 2. operating-point menu
menu = NamedTuple[]
calls = Dict{Tuple{String, String, String}, BitMatrix}()
for (fam, model, _, _) in MODELS, (rname, rule) in RULES
    nc = nested_calls(S[(fam, model)], folds, y, dr, rule)
    calls[(fam, model, rname)] = nc.calls
    rate(m, f) = mean(f(op_metrics(y[m], Float64.(nc.calls[m, r]), 0.5)) for r in axes(nc.calls, 2))
    push!(menu, (family = fam, model = model, rule = rname, t_median = r3(median(filter(isfinite, vec(nc.t)))),
                 dr25_recall = r3(rate(dr, o -> o.recall)), dr25_precision = r3(rate(dr, o -> o.precision)),
                 dr25_fp_kept = r3(rate(dr, o -> o.fp_kept)),
                 fps_kept_per1000 = round(Int, 1000 * rate(dr, o -> o.fp) / sum(dr)),
                 confirmed_lost = r3(rate(dr .& (conf .| (y .== 0)), o -> 1 - o.recall)),
                 supp_recall = r3(rate(.!dr, o -> o.recall)), supp_precision = r3(rate(.!dr, o -> o.precision))))
end
menu = DataFrame(menu); CSV.write(joinpath(CV_DIR, "$(TAG)_operating_points.csv"), menu); show_df(menu)

# 3. blind-spot groups (whole pool and DR25 for NT), NEW vs OLD, paired star-bootstrap CI on the change
a = with_vetting_info(pool, df_raw, zeros(nrow(pool)))
flag(c) = coalesce.(leftjoin(pool[:, [:kepoi_name]], df_raw[:, [:kepoi_name, c]]; on = :kepoi_name, order = :left)[!, c], 0) .== 1
nt_any, ec_any = flag(:koi_fpflag_nt), flag(:koi_fpflag_ec)
fp = y .== 0
groups = [("FP missed: NT only", fp .& (a.flag_pattern .== "NT"), true),
          ("FP missed: NT only (DR25)", fp .& (a.flag_pattern .== "NT") .& dr, true),
          ("FP missed: NT flag (any)", fp .& nt_any, true),
          ("FP missed: EC only", fp .& (a.flag_pattern .== "EC"), true),
          ("FP missed: EC flag (any)", fp .& ec_any, true),
          ("FP missed: CO only", fp .& (a.flag_pattern .== "CO"), true),
          ("FP missed: SS only", fp .& (a.flag_pattern .== "SS"), true),
          ("FP missed: all", fp, true),
          ("CONFIRMED called FP", conf, false),
          ("CONFIRMED called FP (DR25)", conf .& dr, false),
          ("planet called FP: all", y .== 1, false)]
"Per-KOI error rate over fold shuffles (missed FP, or planet called FP)."
errk(C, isfp) = vec(mean(isfp ? C : .!C; dims = 2))
"Paired star bootstrap of mean(e1 − e0) over the group's KOIs (resampling stars)."
function star_boot(e1, e0, stars; B = 2000, rng = Random.Xoshiro(7))
    us = unique(stars); idx = [findall(==(s), stars) for s in us]
    d = e1 .- e0
    bs = map(1:B) do _
        pick = reduce(vcat, idx[rand(rng, 1:length(us), length(us))])
        mean(d[pick])
    end
    return mean(d), quantile(bs, 0.025), quantile(bs, 0.975)
end
crit = NamedTuple[]
for (gname, m, isfp) in groups, fam in ("catalog", "+ centroid"), rname in ("p = 0.5", "planet recall 98%")
    e0 = errk(calls[(fam, OLD, rname)][m, :], isfp)
    e1 = errk(calls[(fam, NEW, rname)][m, :], isfp)
    d, lo, hi = star_boot(e1, e0, pool.kepid[m])
    push!(crit, (group = gname, n = sum(m), family = fam, rule = rname, old_pct = round(100mean(e0); digits = 1),
                 new_pct = round(100mean(e1); digits = 1), delta_pct = round(100d; digits = 1),
                 ci_lo = round(100lo; digits = 1), ci_hi = round(100hi; digits = 1)))
end
println("old = $OLD, new = $NEW")
crit = DataFrame(crit); CSV.write(joinpath(CV_DIR, "$(TAG)_criteria.csv"), crit); show_df(crit)

# 4. low FP rate, DR25 KOIs (koi_score is the robovetter's own output, reference only, never an input)
ks = Float64.(coalesce.(leftjoin(pool[:, [:kepoi_name]], df_raw[:, [:kepoi_name, :koi_score]];
                                 on = :kepoi_name, order = :left).koi_score, NaN))
function recall_at_fpr(yy, s, fpr)
    t = quantile(s[yy .== 0], 1 - fpr)
    return mean(s[yy .== 1] .> t)
end
low = NamedTuple[]
m = dr .& isfinite.(ks)
for fpr in (0.05, 0.10)
    for (fam, model, _, _) in MODELS
        Sm = S[(fam, model)]
        v = [recall_at_fpr(y[m], Sm[m, r], fpr) for r in axes(Sm, 2)]
        push!(low, (fp_rate = fpr, family = fam, model = model, n = sum(m), recall = r3(mean(v)), spread = r3(std(v))))
    end
    push!(low, (fp_rate = fpr, family = "reference", model = "koi_score", n = sum(m),
                recall = r3(recall_at_fpr(y[m], ks[m], fpr)), spread = 0.0))
end
low = DataFrame(low); CSV.write(joinpath(CV_DIR, "$(TAG)_lowfp.csv"), low); show_df(low)
for (fam, model, _, _) in MODELS
    Sm = S[(fam, model)]
    @printf("%-10s %-11s AUC DR25 %.4f, supplemental %.4f (mean over fold shuffles)\n", fam, model,
            mean(fast_auc(y[dr], Sm[dr, r]) for r in axes(Sm, 2)), mean(fast_auc(y[.!dr], Sm[.!dr, r]) for r in axes(Sm, 2)))
end
println(Dates.format(now(), "mm-dd HH:MM"), "  DONE")
