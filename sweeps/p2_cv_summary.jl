# Phase 2, M3: paired comparison of the 50/50 MLP + trees averages (the Phase 1 final recipe) with and without the
# light-curve features, on the standard folds. CV only. Writes sweeps/results/p2_cv_summary.csv.
#   julia --project=. sweeps/p2_cv_summary.jl
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

df_raw = load_koi()
pool, _ = cv_pool(clean_koi(df_raw))
dr = is_dr25(pool, df_raw)
avg(a, b) = (A = load_cv(a); B = load_cv(b); merge(A, (oof = (A.oof .+ B.oof) ./ 2,)))
sub(cv, m) = merge(cv, (oof = cv.oof[m, :, :], folds = cv.folds[m, :], kepoi = cv.kepoi[m]))
fams = [("catalog", avg("cv_mlpq_lc_r5s10.jld2", "cv_evotree_lc_r5s10.jld2"),
         avg("cv_mlp_lr3_e600_quantile_r5s10.jld2", "cv_evotree_tuned_r5s10.jld2")),
        ("catalog + centroid", avg("cv_mlpq_lccent_r5s10.jld2", "cv_evotree_lccent_r5s10.jld2"),
         avg("cv_mlpq_centroid_r5s10.jld2", "cv_evotree_centroid_r5s10.jld2"))]
rows = NamedTuple[]
for (fam, new, old) in fams, (lab, m) in [("DR25 KOIs", dr), ("supplemental KOIs", .!dr), ("all pool", trues(nrow(pool)))]
    c = compare_cv(sub(new, m), sub(old, m), pool[m, :]; B = 1000)
    push!(rows, (family = fam, subset = lab, n = sum(m), phase1 = round(c.B; digits = 4), phase2a = round(c.A; digits = 4),
                 delta = round(c.delta; digits = 4), ci_lo = round(c.ci_lo; digits = 4), ci_hi = round(c.ci_hi; digits = 4)))
end
t = DataFrame(rows)
CSV.write(joinpath(CV_DIR, "p2_cv_summary.csv"), t)
show(stdout, MIME"text/plain"(), t; allcols = true)
println()
