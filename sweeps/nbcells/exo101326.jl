# %% [markdown]
# # Kepler vetting: step 3, baselines (10/13/2026)
#
# **Question:** how does the baseline MLP compare with simpler models on the same 11 catalog features?
# - **Logistic regression** (LR): the same pipeline with the hidden layers removed, a single `Dense(11 => 1)`, with the same
#   scaling, loss and early stopping. Measures how much the MLP's nonlinearity is worth.
# - **Gradient-boosted trees** (EvoTrees.jl, logloss, early-stopped on the same inner validation split as the MLP).
#   The standard strong baseline for tabular data.
#
# All models use the step-1 protocol: the same 5 fold shuffles × 10 seeds on the 8,288-KOI pool as
# `cv_baseline_r5s10.jld2`, compared with `compare_cv` (training-noise null + paired star-bootstrap CI).
# The trees were tuned on a 60-setting grid scored on **different** fold shuffles (`fold_seed0 = 5000`), so the
# final comparison folds were not used for tuning. Compute ran headless in `sweeps/cv_baselines.jl`.
# **The test split is never used here.**
#
# ## Key results
# KEY_RESULTS_PLACEHOLDER

# %%
include(joinpath(@__DIR__, "src", "VettingSetup.jl"))
df_raw = load_koi()
df = clean_koi(df_raw)
pool, _ = cv_pool(df)
y = pool.label
runs = [("MLP (baseline)", load_cv("cv_baseline_r5s10.jld2")),
        ("Logistic regression", load_cv("cv_logreg_r5s10.jld2")),
        ("Trees, untuned", load_cv("cv_evotree_default_r5s10.jld2")),
        ("Trees, tuned", load_cv("cv_evotree_tuned_r5s10.jld2"))]
for (name, cv) in runs
    @assert cv.kepoi == pool.kepoi_name && cv.folds == runs[1][2].folds
end
base = runs[1][2]
scores = Dict(name => oof_scores(cv) for (name, cv) in runs)
println("pool: $(nrow(pool)) KOIs; every run: $(size(base.oof, 2)) fold shuffles × $(size(base.oof, 3)) seeds on identical folds")
println("tuned tree settings: ", runs[4][2].config["train_kw"])

# %% [markdown]
# ## 1. Headline comparison
# Pooled out-of-fold (OOF) scores, each KOI averaged over 50 models that never saw its star. Δ is vs the MLP. A change
# counts as real only if |Δ| clears the training-noise null (`p_null` small) **and** the paired star-bootstrap 95% CI
# excludes 0. For log-loss, lower is better, so a negative Δ is an improvement.

# %%
null_auc = null_differences(base, pool)
null_ll  = null_differences(base, pool; metric = logloss)
rows = map(runs) do (name, cv)
    s = scores[name]
    c  = compare_cv(cv, base, pool; null = null_auc)
    cl = compare_cv(cv, base, pool; null = null_ll, metric = logloss)
    conf = pool.koi_disposition .!= "CANDIDATE"
    (model = name, roc_auc = c.A, d_auc = c.delta, auc_ci = @sprintf("[%+.4f, %+.4f]", c.ci_lo, c.ci_hi),
     p_null = c.p_null, pr_auc = au_prcurve(Int.(y), s), logloss = cl.A, d_logloss = cl.delta,
     ll_ci = @sprintf("[%+.4f, %+.4f]", cl.ci_lo, cl.ci_hi), roc_auc_conf = fast_auc(y[conf], s[conf]))
end
headline = DataFrame(rows)

# %% [markdown]
# ### 1.1 Is the tree gain consistent across folds?
# Each point is one held-out fold (25 = 5 shuffles × 5 folds, ~1,660 KOIs each), scored with the 10-seed ensemble of each
# model. Paired: both models are scored on exactly the same KOIs.

# %%
function fold_auc(cv)
    R, K = size(cv.folds, 2), maximum(cv.folds)
    [fast_auc(y[cv.folds[:, r] .== f], vec(mean(cv.oof[cv.folds[:, r] .== f, r, :]; dims = 2))) for r in 1:R for f in 1:K]
end
fa = Dict(name => fold_auc(cv) for (name, cv) in runs)
for name in ("Trees, untuned", "Trees, tuned", "Logistic regression")
    d = fa[name] .- fa["MLP (baseline)"]
    @printf("%-20s beats MLP on %2d/25 folds; per-fold Δ mean %+.4f, std %.4f, min %+.4f, max %+.4f\n",
            name, sum(d .> 0), mean(d), std(d), minimum(d), maximum(d))
end
fig = Figure(size = (800, 380))
ax = Axis(fig[1, 1]; xlabel = "MLP fold ROC-AUC", ylabel = "tree fold ROC-AUC (tuned)", aspect = 1,
          title = "Per-fold ROC-AUC, trees vs MLP (25 folds)")
scatter!(ax, fa["MLP (baseline)"], fa["Trees, tuned"]; color = :darkgreen, markersize = 10)
lo, hi = extrema([fa["MLP (baseline)"]; fa["Trees, tuned"]])
lines!(ax, [lo, hi], [lo, hi]; color = :gray, linestyle = :dash)
ax2 = Axis(fig[1, 2]; xlabel = "Δ fold ROC-AUC (tuned trees − MLP)", ylabel = "folds")
hist!(ax2, fa["Trees, tuned"] .- fa["MLP (baseline)"]; bins = 12, color = (:darkgreen, 0.6))
vlines!(ax2, [0]; color = :black)
savefig("exo101326_fold_pairs.png", fig)
fig

# %% [markdown]
# ### 1.2 ROC and precision-recall curves (pooled OOF)

# %%
fig = Figure(size = (900, 400))
a1 = Axis(fig[1, 1]; xlabel = "false positive rate", ylabel = "true positive rate", title = "ROC", aspect = 1)
a2 = Axis(fig[1, 2]; xlabel = "recall", ylabel = "precision", title = "Precision-recall", aspect = 1)
for ((name, _), col) in zip(runs, (:steelblue, :gray, :orange, :darkgreen))
    s = scores[name]
    fpr, tpr = roccurve(Int.(y), s)
    rec, prec = prcurve(Int.(y), s)
    lines!(a1, fpr, tpr; color = col, label = @sprintf("%s (%.3f)", name, fast_auc(y, s)))
    lines!(a2, rec, prec; color = col)
end
lines!(a1, [0, 1], [0, 1]; color = :black, linestyle = :dot)
axislegend(a1; position = :rb, labelsize = 11)
ylims!(a2, 0.4, 1.01)
savefig("exo101326_roc_pr.png", fig)
fig

# %% [markdown]
# ## 2. Tree tuning grid
# 60 settings (depth × min_weight × L2 × colsample; eta 0.05, rowsample 0.8, early stopping on the inner validation
# split), each scored with 2 fold shuffles × 3 seeds on **separate** folds (`fold_seed0 = 5000`). The MLP on the same
# tuning folds is the reference line.

# %%
g = JLD2.load(joinpath(CV_DIR, "evotree_grid.jld2"))
gridres, mlp_ref = g["gridres"], g["mlp_ref_auc"]
@printf("MLP on tuning folds: AUC %.4f;  best grid AUC %.4f;  worst %.4f;  untuned default (depth 5, mw 5, L2 1, cs 0.8) is not in the grid\n",
        mlp_ref, maximum(gridres.roc_auc), minimum(gridres.roc_auc))
fig = Figure(size = (900, 380))
for (j, cs) in enumerate((0.5, 0.8))
    ax = Axis(fig[1, j]; xlabel = "max_depth", ylabel = j == 1 ? "pooled OOF ROC-AUC (tuning folds)" : "",
              title = "colsample = $cs", xticks = [3, 4, 5, 6, 8])
    for (mw, l2, col, ls) in ((1.0, 1.0, :steelblue, :solid), (10.0, 1.0, :orange, :solid), (40.0, 1.0, :firebrick, :solid),
                              (1.0, 10.0, :steelblue, :dash), (10.0, 10.0, :orange, :dash), (40.0, 10.0, :firebrick, :dash))
        t = sort(gridres[(gridres.colsample .== cs) .& (gridres.min_weight .== mw) .& (gridres.L2 .== l2), :], :max_depth)
        scatterlines!(ax, t.max_depth, t.roc_auc; color = col, linestyle = ls, label = "min_weight $(Int(mw)), L2 $(Int(l2))")
    end
    hlines!(ax, [mlp_ref]; color = :black, linestyle = :dot, label = "MLP")
    j == 2 && Legend(fig[1, 3], ax; framevisible = false, labelsize = 11)
end
savefig("exo101326_tree_grid.png", fig)
println(first(gridres, 8))
fig

# %% [markdown]
# ## 3. What the models rely on
# One model of each kind fit on a single star-grouped 8/9–1/9 split of the pool (for illustration only; the scores above
# come from the CV). LR coefficients are on z-scored inputs (log10 for period/depth/prad), so their size compares
# directly. Tree importance is the share of total split gain.

# %%
tr, va = split_by_star(pool; at = (8 / 9,), seed = 7)
d1 = prepare_data(tr, va, va)
lrm, = train_model(d1; hidden = (), dropout = 0.0, lr = 1e-2, verbose = false)
tkw = Dict(Symbol(k) => v for (k, v) in runs[4][2].config["train_kw"])
tm, = train_evotree(d1; tkw...)
imp = leftjoin(logreg_coefs(lrm, d1.features), tree_importance(tm); on = :feature)
sort!(imp, :importance; rev = true)

# %% [markdown]
# ## 4. Do the trees fix the step-2 blind spots?
# Same groups as step 2 (threshold 0.5, ensemble OOF scores). Threshold-based rates depend on calibration as well as
# ranking, so group AUCs (by detection source) are shown too.

# %%
a = with_vetting_info(pool, df_raw, scores["MLP (baseline)"])
a.tce_source = String.(coalesce.(a.koi_tce_delivname, "none (no TCE)"))
fp = a.label .== 0
groups = [("FP missed: SS only", fp .& (a.flag_pattern .== "SS")),
          ("FP missed: CO only", fp .& (a.flag_pattern .== "CO")),
          ("FP missed: NT only", fp .& (a.flag_pattern .== "NT")),
          ("FP missed: EC only", fp .& (a.flag_pattern .== "EC")),
          ("FP missed: koi_count ≥ 3", fp .& (a.koi_count .>= 3)),
          ("FP missed: koi_count = 1", fp .& (a.koi_count .== 1)),
          ("FP missed: all", fp),
          ("CONFIRMED called FP", a.koi_disposition .== "CONFIRMED"),
          ("CANDIDATE called FP", a.koi_disposition .== "CANDIDATE"),
          ("planet called FP: P < 1.5 d", (a.label .== 1) .& (a.koi_period .< 1.5)),
          ("planet called FP: all", a.label .== 1)]
err_rate(s, m, isfp) = isfp ? mean(s[m] .>= 0.5) : mean(s[m] .< 0.5)
blind = DataFrame(group = first.(groups), n = [sum(m) for (_, m) in groups])
for (name, _) in runs
    blind[!, name] = [round(100 * err_rate(scores[name], m, startswith(gname, "FP")); digits = 1) for (gname, m) in groups]
end
blind

# %%
src_tbl = combine(groupby(a, :tce_source), nrow => :n)
for (name, _) in runs
    s = scores[name]
    src_tbl[!, name * " AUC"] = [fast_auc(y[a.tce_source .== t], s[a.tce_source .== t]) for t in src_tbl.tce_source]
end
sort!(src_tbl, :n; rev = true)

# %% [markdown]
# ## 5. How different are the MLP and the trees?
# If they make different mistakes, averaging them should help. The 50/50 average uses the same folds and seeds, so it
# goes through the same `compare_cv` test.

# %%
s_mlp, s_tree = scores["MLP (baseline)"], scores["Trees, tuned"]
tuned = runs[4][2]
avg = (oof = (base.oof .+ tuned.oof) ./ 2, folds = base.folds, kepoi = base.kepoi)
c_avg_mlp  = compare_cv(avg, base, pool; null = null_auc)
c_avg_tree = compare_cv(avg, tuned, pool)
@printf("Spearman(MLP, trees) = %.3f;  different hard call on %.1f%% of KOIs\n",
        corspearman(s_mlp, s_tree), 100 * mean((s_mlp .>= 0.5) .!= (s_tree .>= 0.5)))
@printf("MLP+trees average: AUC %.4f;  vs MLP Δ %+.4f [%+.4f, %+.4f] p_null %.3f;  vs trees Δ %+.4f [%+.4f, %+.4f]\n",
        c_avg_mlp.A, c_avg_mlp.delta, c_avg_mlp.ci_lo, c_avg_mlp.ci_hi, c_avg_mlp.p_null,
        c_avg_tree.delta, c_avg_tree.ci_lo, c_avg_tree.ci_hi)
fig = Figure(size = (900, 420))
for (j, (lab, ttl)) in enumerate(((0, "False positives"), (1, "CONFIRMED + CANDIDATE")))
    m = y .== lab
    ax = Axis(fig[1, j]; xlabel = "MLP OOF score", ylabel = j == 1 ? "tree OOF score" : "", title = ttl, aspect = 1)
    h = fit(Histogram, (s_mlp[m], s_tree[m]), (0:0.05:1.0001, 0:0.05:1.0001))
    heatmap!(ax, 0.025:0.05:1, 0.025:0.05:1, log10.(h.weights .+ 1); colormap = :viridis)
    lines!(ax, [0, 1], [0, 1]; color = :white, linestyle = :dash)
    limits!(ax, 0, 1, 0, 1)
end
Colorbar(fig[1, 3]; colormap = :viridis, label = "log10(count + 1)", limits = (0, 3))
savefig("exo101326_mlp_vs_trees.png", fig)
fig

# %% [markdown]
# ### 5.1 Calibration at a glance
# Fraction of planets in 10 score bins (pooled OOF). Full calibration and threshold choice are step 5; this only checks
# whether the threshold-0.5 rates in section 4 compare like with like.

# %%
fig = Figure(size = (520, 440))
ax = Axis(fig[1, 1]; xlabel = "mean predicted score in bin", ylabel = "observed planet fraction", aspect = 1,
          title = "Reliability (10 equal-count bins)")
for ((name, _), col) in zip(runs, (:steelblue, :gray, :orange, :darkgreen))
    s = scores[name]
    edges = quantile(s, range(0, 1; length = 11))
    b = clamp.(searchsortedlast.(Ref(edges), s), 1, 10)
    scatterlines!(ax, [mean(s[b .== i]) for i in 1:10], [mean(y[b .== i]) for i in 1:10]; color = col, label = name)
end
lines!(ax, [0, 1], [0, 1]; color = :black, linestyle = :dot)
axislegend(ax; position = :lt, labelsize = 11)
savefig("exo101326_reliability.png", fig)
fig

# %%
oof_out = DataFrame(kepoi_name = pool.kepoi_name, kepid = pool.kepid, koi_disposition = pool.koi_disposition,
                    label = y, mlp = s_mlp, logreg = scores["Logistic regression"],
                    tree_default = scores["Trees, untuned"], tree_tuned = s_tree)
CSV.write(joinpath(CV_DIR, "oof_baselines_r5s10.csv"), oof_out)
println("wrote sweeps/results/oof_baselines_r5s10.csv  ($(nrow(oof_out)) rows)");

# %% [markdown]
# ## 6. Conclusions
# CONCLUSIONS_PLACEHOLDER
