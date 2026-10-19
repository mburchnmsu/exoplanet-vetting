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
# - **Gradient-boosted trees beat the MLP, and the gain is real.** Tuned trees: pooled OOF ROC-AUC **0.9320** vs **0.9251**
#   (Δ +0.0069, star-bootstrap 95% CI [+0.0045, +0.0094], far outside the 0.0005 training-noise null). They win on
#   **25 of 25** held-out folds, and log-loss improves too (0.331 vs 0.346). Even untuned trees win (+0.0056, 23/25
#   folds), so the conclusion doesn't depend on tuning. CONFIRMED vs FP: 0.973 vs 0.964.
# - **Logistic regression is far behind** (0.867, Δ −0.058, worse on every fold): the MLP's nonlinearity is worth a lot,
#   and trees capture the same structure a bit better.
# - **Tuning matters little except tree depth:** 0.927 at depth 3 → 0.932 at depth 8; min_weight, L2 and colsample move
#   AUC by ≲ 0.001. Depth 8 is the edge of the grid, so deeper trees are untested (likely ≲ 0.0005 more).
# - **The trees fix none of the step-2 blind spots.** Centroid-offset FPs (37.5% missed vs 37.9%), not-transit-like and
#   contamination FPs, FPs in 3+ KOI systems and supplemental KOIs (AUC ≈ 0.58 for Q1–Q16-only) are as bad as for the MLP.
#   The gains are in the parts that were already decent: DR25 KOIs (0.953 vs 0.948), fewer CONFIRMED planets called FP
#   (3.6% vs 4.8%) and ultra-short-period planets (59% called FP vs 70%).
# - **The MLP and the trees mostly agree** (Spearman 0.967; different hard call on 6.5% of KOIs), and averaging them
#   is no better than trees alone (Δ +0.0001, CI includes 0).
# - **Tuning the MLP closes only ~20% of the gap (section 7).** The best of 14 variants (learning rate 3e-3, 600-epoch
#   cap) reaches 0.9265: a real +0.0014 over the baseline, but still 0.0055 behind the trees. Width, depth, dropout,
#   weight decay and batch size don't matter. The likely culprit is how the inputs are scaled, which is step 4 territory.
# - **Step 4, part 1 (section 8): fixing the MLP's inputs closes the gap.**
#   - A rank-gauss (quantile) transform of the features lifts the tuned MLP to **0.9330**: +0.008 over the original MLP,
#     tied with the trees.
#   - Averaging that MLP with the trees gives **0.9352**, a real gain over each alone. This is the new best.
# - **Step 4, part 2 (section 9):**
#   - `koi_count` is worth ~0.004 AUC as a pure multiplicity prior (ranking within each multiplicity group is unchanged
#     without it). Keep it.
#   - `koi_insol` is a rank copy of `koi_teq`, so it adds nothing.
#   - Deeper trees (10, 12) give no gain.
# - **Step 4, part 3 (section 10): centroid features and DR25 scoping.**
#   - Headline metrics are now on DR25-detected KOIs. On those, catalog-only MLP + trees = **0.955**.
#   - Adding 7 centroid features (a separate model, by decision) gives **0.983**, and closes the centroid-offset blind spot
#     (CO-only FPs missed: 36% → 4%).
#   - ⚠ This is partly circular: the robovetter's CO flag thresholds these same numbers.
#   - Training on DR25 KOIs only gives no gain, so we keep training on all KOIs.
# - **Step 5 (section 11): calibration and operating points.**
#   - Both final models are already well calibrated on DR25 KOIs (Platt fit ≈ identity), so no calibrator is used.
#   - Supplemental-KOI probabilities are unreliable, and only a provenance-based fix could help, which is ruled out.
#   - An operating-point menu (recall 95/98/99%, precision 95/99%, max F1, p = 0.5) is frozen with the two final
#     models in `models/step5_frozen.jld2`, ready for the one-time sealed-test evaluation. Example, 98% planet
#     recall on DR25 KOIs: catalog-only keeps 35% of FPs, catalog + centroid 17%.
# - **Sealed test set (section 12, used once): CV is confirmed.** DR25 ROC-AUC: catalog-only **0.957** [0.944, 0.970]
#   (CV 0.954), catalog + centroid **0.988** [0.982, 0.994] (CV 0.983). At the 98% planet-recall threshold, the test
#   recall is 0.988 / 0.993 with 31% / 16% of FPs kept, and no DR25 CONFIRMED planet is lost.
# - **Post-test (section 13): training on CONFIRMED vs FP only** sharpens CONFIRMED-vs-FP ranking a little (+0.004 /
#   +0.001) but is worse at the real task with CANDIDATEs as planets (−0.015 / −0.007). It scores 36–41% of DR25
#   CANDIDATEs as unlike confirmed planets. Not adopted; possibly useful as a secondary score for follow-up priority.
# - **Takeaway:** the catalog-only errors are limited by the *information* in the 11 features, not by the model. The
#   centroid result confirms it: give the model the right information and the biggest blind spot disappears.
#   Bigger wins should come from features (step 4) and light curves (Phase 2) rather than architecture.

# %%
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
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
    local ax = Axis(fig[1, j]; xlabel = "max_depth", ylabel = j == 1 ? "pooled OOF ROC-AUC (tuning folds)" : "",
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
    local ax = Axis(fig[1, j]; xlabel = "MLP OOF score", ylabel = j == 1 ? "tree OOF score" : "", title = ttl, aspect = 1)
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
# 1. **Trees are the stronger tabular model:** +0.007 ROC-AUC and −0.015 log-loss vs the MLP, consistent on every fold,
#    and just as well calibrated (section 5.1). The step-1 bar (0.0005) is cleared by more than 10×.
# 2. **The gap is modest compared with the sampling uncertainty** of the pool itself (±0.006) and the sealed test set
#    (±0.009): on the test split alone, the two models would be hard to tell apart.
# 3. **Same blind spots for every model.** Centroid offsets, not-transit-like signals, contamination, multi-KOI FPs and
#    supplemental KOIs are missed at the same rates by LR, MLP and trees. This points to missing *information*, which
#    is what step 4 (features, supplemental-KOI scoping) and Phase 2 (light curves, centroids) address.
# 4. **Caveat:** the trees were tuned on different fold shuffles of the same pool, so the tuned number is very slightly
#    optimistic. The untuned trees (no tuning at all) already beat the MLP by +0.0056.
#
# **Decisions for the next session (not done here):**
# - Which model to carry forward as the tabular reference for steps 4–5: trees, the MLP, or both?
#   - Trees are cheaper (~0.3 s per model vs ~2 s) and better.
#   - The MLP is the project's stated model and the natural base for the Phase-2 hybrid (catalog features + light-curve CNN).
# - Whether to try closing the MLP gap first (e.g. wider layers, more epochs, since some runs hit the 200-epoch cap;
#   see step 1), or go straight to step 4 features, which should help both.
# - Optional: extend the tree grid to depth 10–12 (≈ 5 min).
#
# **Follow-up (same day, section 7):** we carry both models forward and tried to close the MLP gap by tuning. The best
# MLP (learning rate 3e-3, 600-epoch cap) gains a real +0.0014 but is still 0.0055 behind the trees.
#
# Files: `sweeps/cv_baselines.jl`, `sweeps/cv_mlp_gap.jl` (compute), `sweeps/results/cv_logreg_r5s10.jld2`, `cv_evotree_default_r5s10.jld2`,
# `cv_evotree_tuned_r5s10.jld2`, `evotree_grid.csv/.jld2`, `oof_baselines_r5s10.csv` (per-KOI scores of all models),
# `src/Baselines.jl`, figures `figures/exo101326_*.png`.

# %% [markdown]
# ## 7. Follow-up: can tuning close the MLP gap?
# Decision: carry **both** models forward, and first try to close the MLP's gap by tuning (`sweeps/cv_mlp_gap.jl`).
#
# **Screen:** 11 single changes plus 3 combinations, on the same tuning folds as the tree grid (2 fold shuffles × 3
# seeds). Δ is paired against the unchanged MLP on those folds (0.9255); the best tree there scored 0.9318.
# AdamW weight decay is lr-coupled (decay per step = lr × wd).

# %%
screen = CSV.read(joinpath(CV_DIR, "mlp_screen.csv"), DataFrame)
select(screen, :variant, :roc_auc, :d_auc, :ci_lo, :ci_hi, :logloss, :median_epoch)

# %%
fig = Figure(size = (800, 420))
ax = Axis(fig[1, 1]; xlabel = "pooled OOF ROC-AUC (tuning folds)", yticks = (1:nrow(screen), reverse(screen.variant)),
          title = "MLP variants vs the best tree on the same folds")
order = nrow(screen):-1:1
scatter!(ax, screen.roc_auc, order; color = :steelblue, markersize = 10)
rangebars!(ax, order, screen.roc_auc .- screen.d_auc .+ screen.ci_lo, screen.roc_auc .- screen.d_auc .+ screen.ci_hi;
           direction = :x, color = (:steelblue, 0.5))
vlines!(ax, [screen.roc_auc[screen.variant .== "base"][1]]; color = :gray, linestyle = :dash, label = "MLP baseline (0.9255)")
vlines!(ax, [maximum(gridres.roc_auc)]; color = :darkgreen, linestyle = :dash, label = @sprintf("best tree (%.4f)", maximum(gridres.roc_auc)))
axislegend(ax; position = :rb)
savefig("exo101326_mlp_screen.png", fig)
fig

# %% [markdown]
# Bars: paired 95% star-bootstrap CI of each variant's gain over the baseline, drawn around the variant's AUC.
#
# **Final check on the standard folds** (5 shuffles × 10 seeds, the step-1 protocol) for the best variant, `lr3_e600`
# (learning rate 3e-3, epoch cap 600, patience 50):

# %%
mlp2 = load_cv("cv_mlp_lr3_e600_r5s10.jld2")
@assert mlp2.folds == base.folds
c_b = compare_cv(mlp2, base, pool; null = null_auc)
c_t = compare_cv(mlp2, tuned, pool)
c_l = compare_cv(mlp2, base, pool; null = null_ll, metric = logloss)
@printf("tuned MLP: AUC %.4f, log-loss %.4f\n", c_b.A, c_l.A)
@printf("  vs baseline MLP: Δ AUC %+.4f [%+.4f, %+.4f], p_null %.3f;  Δ log-loss %+.4f [%+.4f, %+.4f]\n",
        c_b.delta, c_b.ci_lo, c_b.ci_hi, c_b.p_null, c_l.delta, c_l.ci_lo, c_l.ci_hi)
@printf("  vs tuned trees:   Δ AUC %+.4f [%+.4f, %+.4f]  (gap closed: %.0f%%)\n",
        c_t.delta, c_t.ci_lo, c_t.ci_hi, 100 * c_b.delta / (c_b.delta - c_t.delta))
@printf("  median best epoch %.0f (baseline %.0f); runs stopping within patience of the cap: %.1f%%\n",
        median(mlp2.results.best_epoch), median(base.results.best_epoch), 100 * mean(mlp2.results.best_epoch .>= 550))

# %% [markdown]
# **Reading this:**
# - **Real but small gains.** A higher learning rate and a 600-epoch cap help, and the gain clears both step-1 checks.
#   Neither change alone, nor more capacity (wider or 3 layers), nor dropout, weight decay or batch size moves the MLP
#   more than ~0.001.
# - **The tuned MLP closes only about a fifth of the gap** to the trees.
# - **The gap is not about training or capacity.** The most likely cause is the **input representation**:
#   - Trees only see the order of each feature's values, so skewed, heavy-tailed inputs don't bother them.
#   - The MLP sees z-scored raw values, and only period, depth and prad are logged.
#   - SNR, duration, teq, srad and koi_count are strongly skewed and are not logged.
#   - Testing that (log SNR etc., or a quantile transform) is a feature-preprocessing change, which belongs in step 4.

# %% [markdown]
# ## 8. Step 4, part 1: input transforms for the MLP
# Decision: adopt the tuned MLP (`lr3_e600`) as the new MLP reference, and test the input-representation explanation.
# As the MLP saw them, several features were extremely skewed: impact (skew 23; values up to 100 against a typical
# 0–1.3), stellar radius (22), duration (6), SNR (5), teq (4); some inputs sat 38σ from the mean. Two fixes, both
# fit on the training part only (`prepare_data(...; transform)`):
# - **`log_more`:** log10 also on duration, SNR (+1), teq, srad, koi_count and impact (+0.1).
# - **`quantile`** (rank-gauss): each feature → its empirical CDF on the training part → standard-normal quantile.
#   Immune to outliers; max |z| drops from 38 to 4.5.
#
# Trees are unaffected by either (they only use the order of values). Same standard folds, 5 shuffles × 10 seeds.

# %%
runs4 = [("MLP baseline (exo100926)", base),
         ("MLP tuned (lr3_e600)", mlp2),
         ("MLP tuned + log_more", load_cv("cv_mlp_lr3_e600_logmore_r5s10.jld2")),
         ("MLP tuned + quantile", load_cv("cv_mlp_lr3_e600_quantile_r5s10.jld2")),
         ("Trees, tuned", tuned)]
for (_, cv) in runs4; @assert cv.folds == base.folds; end
mq = runs4[4][2]
rows4 = map(runs4) do (name, cv)
    cb = compare_cv(cv, base, pool; null = null_auc)
    cm = compare_cv(cv, mlp2, pool)
    ct = compare_cv(cv, tuned, pool)
    s = oof_scores(cv)
    conf = pool.koi_disposition .!= "CANDIDATE"
    (model = name, roc_auc = cb.A, logloss = logloss(y, s), roc_auc_conf = fast_auc(y[conf], s[conf]),
     vs_baseline = @sprintf("%+.4f [%+.4f, %+.4f]", cb.delta, cb.ci_lo, cb.ci_hi),
     vs_tuned_mlp = @sprintf("%+.4f [%+.4f, %+.4f]", cm.delta, cm.ci_lo, cm.ci_hi),
     vs_trees = @sprintf("%+.4f [%+.4f, %+.4f]", ct.delta, ct.ci_lo, ct.ci_hi))
end
DataFrame(rows4)

# %% [markdown]
# ### 8.1 Blind spots and detection sources with the transformed MLP
# Same groups as section 4 (% wrong at threshold 0.5), plus ROC-AUC by TCE source.

# %%
s_q = oof_scores(mq)
blind4 = select(blind, :group, :n, "MLP (baseline)", "Trees, tuned")
blind4[!, "MLP + quantile"] = [round(100 * err_rate(s_q, m, startswith(g, "FP")); digits = 1) for (g, m) in groups]
src4 = select(src_tbl, :tce_source, :n, "MLP (baseline) AUC", "Trees, tuned AUC")
src4[!, "MLP + quantile AUC"] = [fast_auc(y[a.tce_source .== t], s_q[a.tce_source .== t]) for t in src4.tce_source]
println(src4)
blind4

# %% [markdown]
# ### 8.2 Now that they are equally good: do the MLP and the trees combine?

# %%
avg4 = (oof = (mq.oof .+ tuned.oof) ./ 2, folds = base.folds, kepoi = base.kepoi)
c4_q = compare_cv(avg4, mq, pool)
c4_t = compare_cv(avg4, tuned, pool)
@printf("Spearman(quantile MLP, trees) = %.3f;  different hard call on %.1f%% of KOIs\n",
        corspearman(s_q, s_tree), 100 * mean((s_q .>= 0.5) .!= (s_tree .>= 0.5)))
@printf("50/50 average: AUC %.4f, log-loss %.4f;  vs quantile MLP Δ %+.4f [%+.4f, %+.4f];  vs trees Δ %+.4f [%+.4f, %+.4f]\n",
        c4_q.A, logloss(y, oof_scores(avg4)), c4_q.delta, c4_q.ci_lo, c4_q.ci_hi, c4_t.delta, c4_t.ci_lo, c4_t.ci_hi)
fig = Figure(size = (900, 400))
a1 = Axis(fig[1, 1]; xlabel = "false positive rate", ylabel = "true positive rate", title = "ROC (pooled OOF)", aspect = 1)
a2 = Axis(fig[1, 2]; xlabel = "recall", ylabel = "precision", title = "Precision-recall", aspect = 1)
for ((name, sc), col) in zip((("MLP baseline", s_mlp), ("MLP + quantile", s_q), ("Trees, tuned", s_tree),
                              ("average", oof_scores(avg4))), (:gray, :steelblue, :darkgreen, :purple))
    fpr, tpr = roccurve(Int.(y), sc); rec, prec = prcurve(Int.(y), sc)
    lines!(a1, fpr, tpr; color = col, label = @sprintf("%s (%.4f)", name, fast_auc(y, sc)))
    lines!(a2, rec, prec; color = col)
end
axislegend(a1; position = :rb, labelsize = 11)
xlims!(a1, 0, 0.5); ylims!(a1, 0.5, 1.0); ylims!(a2, 0.6, 1.01)
savefig("exo101326_transforms_roc.png", fig)
fig

# %% [markdown]
# **Reading section 8:**
# - **The input representation was the whole gap.**
#   - The quantile-transformed MLP reaches **0.9330**: +0.008 over the original MLP, +0.0065 over the tuned MLP.
#   - That's **level with the tuned trees** (Δ +0.0010, CI includes 0).
#   - `log_more` gets almost as far (0.9326). The rest of the tuning (width, depth, dropout…) never mattered.
# - **Blind spots shrink a little but remain** (threshold 0.5):
#
#   | Group | MLP baseline | Quantile MLP |
#   |---|---|---|
#   | Missed FPs overall | 16.1% | 14.3% |
#   | NT-only FPs missed | 24.6% | 20.4% |
#   | CO-only FPs missed | 37.9% | 35.1% |
#   | FPs with 3+ KOIs missed | 79% | 74% |
#   | Q1–Q16-only KOIs, AUC | 0.56 | 0.58 |
#
#   CANDIDATEs called FP rise slightly (31% → 33%), partly a threshold shift. Centroid offsets and supplemental KOIs
#   are still the limit: an information problem, not a model problem.
# - **Now the two models do combine.** The quantile MLP and the trees agree slightly more than before (Spearman 0.976),
#   yet the 50/50 average reaches **0.9352**:
#   - +0.0022 [+0.0011, +0.0032] over the quantile MLP;
#   - +0.0032 [+0.0021, +0.0043] over the trees;
#   - log-loss 0.324, the best so far.
#
#   With the original MLP, averaging added nothing. Once both models are equally good, their remaining errors are partly independent.
# - **Recommendation for the rest of step 4:**
#   - MLP reference: `transform = :quantile` + `lr3_e600`.
#   - Keep the trees.
#   - Report the MLP + trees average as the combined model.

# %% [markdown]
# ## 9. Step 4, part 2: `koi_count` ablation, `koi_insol`, deeper trees
# Compute: `sweeps/step4_features.jl`. References: quantile MLP, tuned trees, and their average (section 8).
#
# **`koi_insol` (not run, on purpose):** on the pool, Spearman(`koi_insol`, `koi_teq`) = **1.0000**, because teq is computed
# from insolation. Trees only use the order of values, and the quantile transform maps both to identical inputs, so
# `koi_insol` adds exactly zero information to either reference model. It stays out.
#
# **`koi_count` ablation:** how much of the performance is the multiplicity prior ("a star with several KOIs is
# probably a planetary system")? Both models retrained on the 10 other features, same folds.

# %%
mq_nc = load_cv("cv_mlpq_nocount_r5s10.jld2")
tr_nc = load_cv("cv_evotree_nocount_r5s10.jld2")
avg_nc = (oof = (mq_nc.oof .+ tr_nc.oof) ./ 2, folds = base.folds, kepoi = base.kepoi)
abl = map([("MLP + quantile", mq, mq_nc), ("Trees, tuned", tuned, tr_nc), ("MLP + trees average", avg4, avg_nc)]) do (name, with, without)
    c = compare_cv(without, with, pool)
    cl = compare_cv(without, with, pool; metric = logloss)
    (model = name, with_count = c.B, without_count = c.A, d_auc = @sprintf("%+.4f [%+.4f, %+.4f]", c.delta, c.ci_lo, c.ci_hi),
     d_logloss = @sprintf("%+.4f [%+.4f, %+.4f]", cl.delta, cl.ci_lo, cl.ci_hi))
end
DataFrame(abl)

# %% [markdown]
# Where does the loss come from? Error rates (threshold 0.5) for the average model, with and without `koi_count`,
# split by multiplicity:

# %%
s_av, s_av_nc = oof_scores(avg4), oof_scores(avg_nc)
mult_groups = [("FP missed: koi_count = 1", fp .& (a.koi_count .== 1)),
               ("FP missed: koi_count = 2", fp .& (a.koi_count .== 2)),
               ("FP missed: koi_count ≥ 3", fp .& (a.koi_count .>= 3)),
               ("planet called FP: koi_count = 1", (y .== 1) .& (a.koi_count .== 1)),
               ("planet called FP: koi_count = 2", (y .== 1) .& (a.koi_count .== 2)),
               ("planet called FP: koi_count ≥ 3", (y .== 1) .& (a.koi_count .>= 3))]
DataFrame(group = first.(mult_groups), n = [sum(m) for (_, m) in mult_groups],
          with_count = [round(100 * err_rate(s_av, m, startswith(g, "FP")); digits = 1) for (g, m) in mult_groups],
          without_count = [round(100 * err_rate(s_av_nc, m, startswith(g, "FP")); digits = 1) for (g, m) in mult_groups])

# %%
multi_auc = DataFrame(koi_count = String[], n = Int[], planet_frac = Float64[], auc_with = Float64[], auc_without = Float64[])
for (lab, m) in (("1", a.koi_count .== 1), ("2", a.koi_count .== 2), ("≥ 3", a.koi_count .>= 3))
    push!(multi_auc, (lab, sum(m), mean(y[m]), fast_auc(y[m], s_av[m]), fast_auc(y[m], s_av_nc[m])))
end
multi_auc

# %% [markdown]
# **Deeper trees:** depth 8 was the edge of the step-3 grid. Screened on the same tuning folds (min_weight 10, L2 10,
# colsample 0.8):

# %%
CSV.read(joinpath(CV_DIR, "evotree_depth_screen.csv"), DataFrame)

# %% [markdown]
# **Reading section 9:**
# - **`koi_count` is worth ~0.004 ROC-AUC to every model** (MLP −0.0036, trees −0.0040, average −0.0037; all CIs
#   exclude 0; log-loss worsens by ~0.009).
# - **It is a pure prior: it works between groups, not within them.**
#   - Within each multiplicity group, ranking quality is unchanged without it (AUC 0.912 vs 0.912 for single-KOI stars,
#     0.921 vs 0.923 for 2, 0.908 vs 0.907 for 3+).
#   - All it does is shift scores by group: multi-KOI systems are 81% (2 KOIs) and 95% (3+) planets, singles only 39%.
#   - This matches the known multiplicity boost (false positives rarely come in multiples), and is a legitimate prior,
#     not leakage.
# - **The trade-off is visible at threshold 0.5** (average model, with → without `koi_count`):
#
#   | Group | With | Without |
#   |---|---|---|
#   | FPs missed, 3+ KOI systems | 77% | 51% |
#   | FPs missed, 2-KOI systems | 37% | 27% |
#   | Planets called FP, 2-KOI systems | 2.9% | 6.2% |
#   | Planets called FP, 3+ KOI systems | 1.2% | 3.8% |
#
#   Since multi-KOI systems are mostly planets, the net effect is worse.
# - **Recommendation: keep `koi_count`.** Remember that the model "trusts" multis by design: a false positive on a
#   multi-planet star is the hardest case for the catalog-only model, and an obvious target for Phase 2.
# - **`koi_insol`: stays out** (a rank copy of teq; zero new information).
# - **Deeper trees: no gain.** Depth 10: +0.0001, depth 12: +0.0003 over depth 8 on the tuning folds, below the 0.0005
#   bar. Depth 8 stays.

# %% [markdown]
# ## 10. Step 4, part 3: centroid features and supplemental-KOI scoping
# **Decisions (10/13), from the recommendations:**
# 1. **Centroid features are allowed, as a separate model.**
#    - The "catalog + centroid" model sits next to the catalog-only reference; both are reported.
#    - The 7 features (`src/Centroids.jl`): difference-image offsets from the out-of-transit centroid and from the
#      KIC position, in arcsec and in σ, plus flux-weighted centroid motion (significance, source offset, in-transit shift).
#    - Missing values (5–10%) get the training median. There are deliberately **no missing-value indicators**: missingness
#      is informative (17% planets vs 52%) and total for no-TCE KOIs, so an indicator would smuggle in provenance.
#    - **Caveat (circularity):** the robovetter's CO flag thresholds these same numbers, so gains on CO false positives
#      partly re-learn its rule.
# 2. **Supplemental KOIs stay in training.**
#    - Headline metrics are now on **DR25-detected KOIs**; supplemental KOIs are reported separately.
#    - Provenance is never a model input.
#    - Tested here: is training on DR25 KOIs only any better for DR25 KOIs?
#
# Compute: `sweeps/step4_scoping.jl`. All runs use the standard folds, 5 × 10, paired.

# %%
dr = is_dr25(pool, df_raw)
pool_c = add_centroid_features(pool, df_raw)
has_cent = isfinite.(pool_c.cent_dicco_msky) .& isfinite.(pool_c.cent_fwm_stat_sig)
avg_of(c1, c2) = (oof = (c1.oof .+ c2.oof) ./ 2, folds = c1.folds, kepoi = c1.kepoi)
mq_c, tr_c = load_cv("cv_mlpq_centroid_r5s10.jld2"), load_cv("cv_evotree_centroid_r5s10.jld2")
mq_d, tr_d = load_cv("cv_mlpq_dr25train_r5s10.jld2"), load_cv("cv_evotree_dr25train_r5s10.jld2")
avg_c, avg_d = avg_of(mq_c, tr_c), avg_of(mq_d, tr_d)
# paired comparison restricted to a subset of KOIs (bootstrap over the stars in the subset)
sub(cv, m) = (oof = cv.oof[m, :, :], folds = cv.folds[m, :], kepoi = cv.kepoi[m])
cmp_on(A, B, m) = compare_cv(sub(A, m), sub(B, m), pool[m, :]; B = 1000)
subsets = [("all KOIs", trues(nrow(pool))), ("DR25 KOIs (headline)", dr), ("supplemental KOIs", .!dr),
           ("KOIs with centroid data", has_cent)]
@printf("DR25 KOIs: %d;  supplemental: %d;  with centroid data: %d (%.1f%% of DR25 KOIs)\n",
        sum(dr), sum(.!dr), sum(has_cent), 100 * mean(has_cent[dr]))

# %% [markdown]
# ### 10.1 Catalog + centroid vs catalog-only
# Δ = (catalog + centroid) − (catalog-only), paired star-bootstrap 95% CI, per subset.

# %%
cent_rows = NamedTuple[]
for (mname, A, B) in (("MLP + quantile", mq_c, mq), ("Trees, tuned", tr_c, tuned), ("MLP + trees average", avg_c, avg4))
    for (sname, m) in subsets
        c = cmp_on(A, B, m)
        push!(cent_rows, (model = mname, subset = sname, n = sum(m), catalog_only = c.B, with_centroid = c.A,
                          delta = @sprintf("%+.4f [%+.4f, %+.4f]", c.delta, c.ci_lo, c.ci_hi)))
    end
end
DataFrame(cent_rows)

# %% [markdown]
# Does it fix the centroid-offset blind spot? Error rates (threshold 0.5) for the average model:

# %%
s_c = oof_scores(avg_c)
cent_groups = [groups; ("FP missed: CO and/or EC", fp .& ((a.koi_fpflag_co .== 1) .| (a.koi_fpflag_ec .== 1)))]
DataFrame(group = first.(cent_groups), n = [sum(m) for (_, m) in cent_groups],
          catalog_only = [round(100 * err_rate(s_av, m, startswith(g, "FP")); digits = 1) for (g, m) in cent_groups],
          with_centroid = [round(100 * err_rate(s_c, m, startswith(g, "FP")); digits = 1) for (g, m) in cent_groups])

# %%
fig = Figure(size = (900, 400))
for (j, (t, m)) in enumerate((("CO-only false positives (n = $(sum(fp .& (a.flag_pattern .== "CO"))))", fp .& (a.flag_pattern .== "CO")),
                               ("Planets (CONFIRMED + CANDIDATE)", y .== 1)))
    local ax = Axis(fig[1, j]; xlabel = "planet score (average model)", ylabel = j == 1 ? "KOIs" : "", title = t)
    hist!(ax, s_av[m]; bins = 0:0.05:1, color = (:gray, 0.5), label = "catalog-only")
    hist!(ax, s_c[m]; bins = 0:0.05:1, color = (:purple, 0.4), label = "catalog + centroid")
    vlines!(ax, [0.5]; color = :black, linestyle = :dash)
    j == 1 && axislegend(ax; position = :ct)
end
savefig("exo101326_centroid_scores.png", fig)
fig

# %% [markdown]
# ### 10.2 Supplemental KOIs: train on all KOIs, or on DR25 KOIs only?
# Scored on DR25 KOIs (the headline set) and on supplemental KOIs. Δ = (DR25-only training) − (training on all).

# %%
scope_rows = NamedTuple[]
for (mname, A, B) in (("MLP + quantile", mq_d, mq), ("Trees, tuned", tr_d, tuned), ("MLP + trees average", avg_d, avg4))
    for (sname, m) in subsets[2:3]
        c = cmp_on(A, B, m)
        push!(scope_rows, (model = mname, scored_on = sname, train_all = c.B, train_dr25_only = c.A,
                           delta = @sprintf("%+.4f [%+.4f, %+.4f]", c.delta, c.ci_lo, c.ci_hi)))
    end
end
DataFrame(scope_rows)

# %% [markdown]
# ### 10.3 New headline table: DR25-detected KOIs
# Every model scored on the 7,199 DR25 KOIs of the pool (supplemental KOIs reported alongside). The robovetter's
# `koi_score` is shown for reference only: it is partly circular with the labels and uses pixel-level tests.

# %%
ks = coalesce.(a.koi_score, NaN)
headline_models = [("MLP baseline (exo100926)", base), ("MLP + quantile", mq), ("Trees, tuned", tuned),
                   ("MLP + trees average", avg4), ("Average, catalog + centroid", avg_c)]
hl = DataFrame(model = String[], dr25_roc_auc = Float64[], dr25_ci = String[], dr25_conf_vs_fp = Float64[], supplemental_roc_auc = Float64[])
for (name, cv) in headline_models
    s = oof_scores(cv)
    bs = star_bootstrap(y[dr], s[dr], pool.kepid[dr]; B = 1000)
    conf = dr .& (pool.koi_disposition .!= "CANDIDATE")
    push!(hl, (name, fast_auc(y[dr], s[dr]), @sprintf("[%.4f, %.4f]", bs.lo, bs.hi), fast_auc(y[conf], s[conf]),
               fast_auc(y[.!dr], s[.!dr])))
end
mk = dr .& isfinite.(ks)
push!(hl, ("robovetter koi_score (reference)", fast_auc(y[mk], ks[mk]), "—",
           fast_auc(y[mk .& (pool.koi_disposition .!= "CANDIDATE")], ks[mk .& (pool.koi_disposition .!= "CANDIDATE")]), NaN))
hl

# %% [markdown]
# **Reading section 10:**
# - **Centroid features are by far the biggest single gain so far.**
#   - Average model on DR25 KOIs: 0.955 → **0.983** (Δ +0.028 [+0.024, +0.032]); all KOIs: 0.935 → 0.970.
#   - Supplemental KOIs: 0.67 → 0.75.
#   - **Not a missing-data artifact:** the gain is the same (+0.034) on KOIs that actually have centroid measurements.
# - **The centroid-offset blind spot is essentially closed.**
#
#   | Average model, threshold 0.5 | Catalog-only | + centroid |
#   |---|---|---|
#   | CO-only FPs missed | 36% | **4.3%** |
#   | CO/EC FPs missed | 19% | 5% |
#   | All FPs missed | 15% | 10% |
#   | CONFIRMED called FP | 3.6% | 1.4% |
#   | Planets with P < 1.5 d called FP | 58% | 32% |
#
#   USP planets probably gain because clean centroids vouch for small, on-target signals.
#   Not-transit-like and contamination FPs are not helped: NT 22% → 27%, EC 23% → 25%, partly a threshold shift.
# - **⚠ Circularity: read 0.983 with care.**
#   - The robovetter set the CO flag (and many FP dispositions) by thresholding these same centroid statistics.
#   - Part of the gain is the model re-learning that rule, which is why it now even out-ranks the robovetter's own
#     `koi_score` on DR25 KOIs (0.983 vs 0.970; koi_score was never built to maximise AUC).
#   - **The catalog-only model (0.955 on DR25) remains the "independent" reference.** The centroid model is the
#     "best use of available DV diagnostics" reference. Report both, always side by side.
# - **Supplemental KOIs: keep training on everything.**
#   - Training on DR25 KOIs only changes nothing for DR25 KOIs (Δ between −0.0002 and +0.0007, all CIs include 0).
#   - It makes supplemental KOIs clearly worse (−0.03 to −0.04).
#   - So the plan stands: train on all KOIs, headline on DR25 KOIs, report supplemental KOIs separately.
# - **New headline numbers (DR25 KOIs, pooled OOF):**
#
#   | Model | ROC-AUC |
#   |---|---|
#   | Original MLP | 0.948 |
#   | Catalog-only MLP + trees | **0.955** [0.951, 0.960] |
#   | Catalog + centroid | **0.983** [0.981, 0.985] |
#   | Robovetter `koi_score` (reference) | 0.970 |

# %% [markdown]
# ## 11. Step 5: calibration and operating points
# **Questions:** are the two final models' scores honest probabilities, and which decision thresholds are on offer?
# - **Models:** catalog-only (quantile MLP + tuned trees, 50/50 average) and catalog + centroid (the same, with the
#   7 centroid features). Both always reported side by side.
# - **Per-repeat scores.** The model we would ship is ONE 10-seed ensemble, so calibration uses each fold shuffle's
#   seed-mean separately. (The repeat-mean used for AUC above is a 50-model ensemble: smoother than anything shipped.)
# - **Calibrators compared:** none, Platt (logistic on logit score) and isotonic.
# - **Nested, so the numbers are honest:** for each held-out fold, the calibrator and every threshold are fitted on
#   the *other* folds' out-of-fold scores, then applied to the held-out fold.
# - **Thresholds are picked on DR25 KOIs** (the headline set), with one threshold for every KOI: provenance is never
#   used to set a KOI's threshold. Supplemental KOIs are reported separately at the same threshold.
# - **User decisions (10/13):** report a menu of operating points rather than one; open the sealed test set later,
#   together. The frozen models are made by `sweeps/step5_freeze.jl` (`models/step5_frozen.jld2`).

# %%
yf = Float64.(y)
conf = pool.koi_disposition .== "CONFIRMED"
final_models = [("catalog-only", (repeat_scores(mq) .+ repeat_scores(tuned)) ./ 2),
                ("catalog + centroid", (repeat_scores(mq_c) .+ repeat_scores(tr_c)) ./ 2)]
folds5 = base.folds
cal_subsets = [("DR25 KOIs", dr), ("supplemental KOIs", .!dr), ("all KOIs", trues(length(yf)))]
cal_rows = NamedTuple[]
for (mname, S) in final_models, meth in (:none, :platt, :isotonic)
    P = nested_calibrate(S, folds5, yf, meth)
    for (sname, m) in cal_subsets
        mean_r(f) = mean(f(yf[m], P[m, r]) for r in axes(P, 2))     # metric averaged over fold shuffles
        push!(cal_rows, (model = mname, calibrator = string(meth), subset = sname, logloss = round(mean_r(logloss); digits = 4),
                         brier = round(mean_r(brier); digits = 4), ece = round(mean_r(ece); digits = 4),
                         roc_auc = round(mean_r(fast_auc); digits = 4), mean_p = round(mean(P[m, :]); digits = 3),
                         planet_rate = round(mean(yf[m]); digits = 3)))
    end
end
for (mname, S) in final_models
    c = fit_platt(vec(S), repeat(yf, size(S, 2)))
    @printf("%s: Platt fit on all OOF scores: a = %.3f, b = %+.3f  (a = 1, b = 0 means already calibrated)\n", mname, c.a, c.b)
end
DataFrame(cal_rows)

# %% [markdown]
# Reliability per subset (uncalibrated, 15 equal-count bins, pooled over the 5 fold shuffles):

# %%
fig = Figure(size = (900, 440))
for (j, (sname, m)) in enumerate(cal_subsets[1:2])
    local ax = Axis(fig[1, j]; xlabel = "predicted planet probability", ylabel = j == 1 ? "observed planet fraction" : "",
                    title = sname, aspect = 1)
    for ((mname, S), col) in zip(final_models, (:steelblue, :purple))
        s, t = vec(S[m, :]), repeat(yf[m], size(S, 2))
        o = sortperm(s); e = round.(Int, range(0, length(s); length = 16))
        scatterlines!(ax, [mean(s[o[e[i]+1:e[i+1]]]) for i in 1:15], [mean(t[o[e[i]+1:e[i+1]]]) for i in 1:15];
                      color = col, label = mname)
    end
    lines!(ax, [0, 1], [0, 1]; color = :black, linestyle = :dot)
    j == 1 && axislegend(ax; position = :lt)
end
savefig("exo101326_step5_reliability.png", fig)
fig

# %% [markdown]
# ### 11.1 Operating-point menu
# Each rule picks a threshold (call "planet" if score ≥ t). Rates are nested estimates (threshold chosen without the
# held-out fold), averaged over the 5 fold shuffles. *Frozen t* is the threshold stored with the final model (median
# over fold shuffles, chosen on all pool DR25 KOIs). Per-1000 columns: out of 1,000 DR25 KOIs with this pool's mix
# (51% planets).

# %%
RULES5 = [("planet recall 95%", (:recall, 0.95)), ("planet recall 98%", (:recall, 0.98)),
          ("planet recall 99%", (:recall, 0.99)), ("precision 95%", (:precision, 0.95)),
          ("precision 99%", (:precision, 0.99)), ("max F1", (:maxf1,)), ("p = 0.5", (:prob, 0.5))]
frozen = JLD2.load(joinpath(@__DIR__, "..", "models", "step5_frozen.jld2"))["frozen"]
menu = NamedTuple[]
for ((mname, S), key) in zip(final_models, ("catalog_only", "catalog_centroid"))
    for (rname, rule) in RULES5
        nc = nested_calls(S, folds5, yf, dr, rule)
        rate(m, f) = mean(f(op_metrics(yf[m], Float64.(nc.calls[m, r]), 0.5)) for r in axes(S, 2))
        n_dr = sum(dr)
        push!(menu, (model = mname, rule = rname, frozen_t = round(frozen[key]["thresholds"][string(rule)]; digits = 3),
                     t_spread = round(std(nc.t); digits = 3),
                     dr25_recall = round(rate(dr, o -> o.recall); digits = 3),
                     dr25_precision = round(rate(dr, o -> o.precision); digits = 3),
                     dr25_fp_kept = round(rate(dr, o -> o.fp_kept); digits = 3),
                     planets_lost_per1000 = round(Int, 1000 * rate(dr, o -> o.fn) / n_dr),
                     fps_kept_per1000 = round(Int, 1000 * rate(dr, o -> o.fp) / n_dr),
                     confirmed_lost = round(rate(dr .& (conf .| (yf .== 0)), o -> 1 - o.recall); digits = 3),
                     supp_recall = round(rate(.!dr, o -> o.recall); digits = 3),
                     supp_precision = round(rate(.!dr, o -> o.precision); digits = 3)))
    end
end
menu_df = DataFrame(menu)
CSV.write(joinpath(CV_DIR, "step5_operating_points.csv"), menu_df)
menu_df

# %% [markdown]
# The trade-off curve on DR25 KOIs (nested-free view: pooled uncalibrated scores of all 5 fold shuffles), with the
# menu's frozen thresholds marked:

# %%
fig = Figure(size = (900, 420))
for (j, ((mname, S), key)) in enumerate(zip(final_models, ("catalog_only", "catalog_centroid")))
    local ax = Axis(fig[1, j]; xlabel = "FPs kept (fraction of DR25 FPs called planet)",
                    ylabel = j == 1 ? "planet recall (DR25)" : "", title = mname, limits = (0, 0.6, 0.6, 1.0))
    s, t = vec(S[dr, :]), repeat(yf[dr], size(S, 2))
    ts = range(0, 1; length = 401)
    lines!(ax, [mean(s[t .== 0] .>= u) for u in ts], [mean(s[t .== 1] .>= u) for u in ts]; color = :gray)
    for ((rname, rule), col) in zip(RULES5, Makie.wong_colors())
        u = frozen[key]["thresholds"][string(rule)]
        scatter!(ax, [mean(s[t .== 0] .>= u)], [mean(s[t .== 1] .>= u)]; color = col, markersize = 12,
                 label = @sprintf("%s (t = %.2f)", rname, u))
    end
    axislegend(ax; position = :rb, labelsize = 10)
end
savefig("exo101326_step5_menu.png", fig)
fig

# %% [markdown]
# **Reading section 11:**
# - **Both models are already well calibrated; no calibrator is needed.**
#   - A Platt fit on all OOF scores is almost the identity (a ≈ 1.04–1.05, b ≈ −0.02).
#   - DR25 KOIs: ECE 0.018 (catalog-only) and 0.014 (centroid) uncalibrated. Platt and isotonic shave ECE by
#     < 0.006 but don't improve log-loss or Brier (isotonic makes log-loss slightly *worse* and costs a little AUC
#     through ties).
#   - Frozen choice: **no calibrator**. The scores can be read as planet probabilities on DR25-like KOIs.
# - **Supplemental KOIs are miscalibrated (ECE 0.05–0.08), and calibration can't fix it.**
#   - The cause is weak ranking (AUC 0.66 / 0.75), not a shifted base rate (mean p 0.45 vs rate 0.43).
#   - Fixing it would need a provenance-specific calibrator, i.e. using provenance as an input. Ruled out by decision.
#   - So supplemental-KOI probabilities should be read as **unreliable**, and the menu reports them separately.
# - **The thresholds are stable and the nested targets are hit** (DR25 recall 0.950 / 0.980 / 0.990 as asked;
#   threshold spread across fold shuffles ≤ 0.04).
# - **Centroid features buy a much better trade-off at every operating point**, for example at 98% planet recall:
#   catalog-only keeps 35% of DR25 FPs, catalog + centroid 17%.
# - **Nothing here touched the test split.** The frozen models + this menu are what the one-time test evaluation will
#   score.

# %% [markdown]
# ## 12. One-time sealed-test evaluation
# - **The test split** (`split_by_star` seed 42, about 10% of stars) was scored exactly once by `sweeps/step5_test.jl`.
#   It used the frozen models from section 11, with no fitting or tuning, and it refuses to run a second time.
# - **Headline (user decision, 10/13):** ROC-AUC on DR25 KOIs, and the **98% planet recall** operating point. Best F1
#   is shown alongside, and the full menu below.
# - **"CV expected"** is the honest cross-validation estimate for the same quantity. For AUC, it's a single 10-seed
#   ensemble per fold shuffle, averaged over shuffles. For operating points, it's the nested estimates from section 11.
# - **The test set is small** (913 KOIs, 792 stars), so its confidence intervals (star bootstrap) are about 3× wider
#   than the CV ones. The question is whether CV and test *agree*, not which is higher.

# %%
tp5 = CSV.read(joinpath(CV_DIR, "test_predictions.csv"), DataFrame)
yt, drt = Float64.(tp5.label), Bool.(tp5.dr25)
conft = tp5.koi_disposition .== "CONFIRMED"
@printf("test: %d KOIs, %d stars;  DR25 %d, supplemental %d;  planet rate %.3f\n",
        nrow(tp5), length(unique(tp5.kepid)), sum(drt), sum(.!drt), mean(yt))
test_models = [("catalog-only", tp5.p_catalog, final_models[1][2]), ("catalog + centroid", tp5.p_centroid, final_models[2][2])]
auc_rows = NamedTuple[]
for (mname, p, S) in test_models
    for (sname, mt, mc) in (("DR25 KOIs (headline)", drt, dr), ("DR25 CONFIRMED vs FP", drt .& (conft .| (yt .== 0)), dr .& (conf .| (yf .== 0))),
                            ("supplemental KOIs", .!drt, .!dr), ("all KOIs", trues(length(yt)), trues(length(yf))))
        bs = star_bootstrap(yt[mt], p[mt], tp5.kepid[mt]; B = 2000)
        cv_exp = mean(fast_auc(yf[mc], S[mc, r]) for r in axes(S, 2))
        push!(auc_rows, (model = mname, subset = sname, n = sum(mt), test_auc = round(fast_auc(yt[mt], p[mt]); digits = 4),
                         test_ci = @sprintf("[%.3f, %.3f]", bs.lo, bs.hi), cv_expected = round(cv_exp; digits = 4)))
    end
end
mk = drt .& isfinite.(tp5.koi_score)
push!(auc_rows, (model = "robovetter koi_score (reference)", subset = "DR25 KOIs (headline)", n = sum(mk),
                 test_auc = round(fast_auc(yt[mk], tp5.koi_score[mk]); digits = 4), test_ci = "—", cv_expected = NaN))
test_auc = DataFrame(auc_rows)
CSV.write(joinpath(CV_DIR, "test_auc.csv"), test_auc)
test_auc

# %% [markdown]
# ### 12.1 Operating points on the test set (frozen thresholds)

# %%
cvmenu = CSV.read(joinpath(CV_DIR, "step5_operating_points.csv"), DataFrame)
op_rows = NamedTuple[]
for ((mname, p, _), key) in zip(test_models, ("catalog_only", "catalog_centroid")), (rname, rule) in RULES5
    t = frozen[key]["thresholds"][string(rule)]
    o = op_metrics(yt[drt], p[drt], t)
    rec_bs = star_bootstrap(yt[drt], p[drt], tp5.kepid[drt]; B = 1000, metric = (a, b) -> op_metrics(a, b, t).recall)
    prec_bs = star_bootstrap(yt[drt], p[drt], tp5.kepid[drt]; B = 1000, metric = (a, b) -> op_metrics(a, b, t).precision)
    cvr = cvmenu[(cvmenu.model .== mname) .& (cvmenu.rule .== rname), :]
    oc = op_metrics(yt[drt .& (conft .| (yt .== 0))], p[drt .& (conft .| (yt .== 0))], t)
    os = op_metrics(yt[.!drt], p[.!drt], t)
    push!(op_rows, (model = mname, rule = rname, t = round(t; digits = 3),
                    recall = @sprintf("%.3f [%.3f, %.3f]", o.recall, rec_bs.lo, rec_bs.hi), cv_recall = cvr.dr25_recall[1],
                    precision = @sprintf("%.3f [%.3f, %.3f]", o.precision, prec_bs.lo, prec_bs.hi), cv_precision = cvr.dr25_precision[1],
                    fp_kept = round(o.fp_kept; digits = 3), cv_fp_kept = cvr.dr25_fp_kept[1],
                    confirmed_lost = round(1 - oc.recall; digits = 3), cv_confirmed_lost = cvr.confirmed_lost[1],
                    supp_recall = round(os.recall; digits = 2), supp_precision = round(os.precision; digits = 2)))
end
test_ops = DataFrame(op_rows)
CSV.write(joinpath(CV_DIR, "test_operating_points.csv"), test_ops)
test_ops

# %%
fig = Figure(size = (900, 420))
a1 = Axis(fig[1, 1]; xlabel = "false positive rate", ylabel = "true positive rate", title = "Test ROC, DR25 KOIs", aspect = 1)
a2 = Axis(fig[1, 2]; xlabel = "predicted planet probability", ylabel = "observed planet fraction",
          title = "Test reliability, DR25 KOIs (8 bins)", aspect = 1)
for ((mname, p, _), col) in zip(test_models, (:steelblue, :purple))
    fpr, tpr = roccurve(Int.(yt[drt]), p[drt])
    lines!(a1, fpr, tpr; color = col, label = @sprintf("%s (%.3f)", mname, fast_auc(yt[drt], p[drt])))
    t98 = frozen[mname == "catalog-only" ? "catalog_only" : "catalog_centroid"]["thresholds"]["(:recall, 0.98)"]
    o = op_metrics(yt[drt], p[drt], t98)
    scatter!(a1, [o.fp_kept], [o.recall]; color = col, markersize = 12)
    s, tt = p[drt], yt[drt]; ord = sortperm(s); e = round.(Int, range(0, length(s); length = 9))
    scatterlines!(a2, [mean(s[ord[e[i]+1:e[i+1]]]) for i in 1:8], [mean(tt[ord[e[i]+1:e[i+1]]]) for i in 1:8]; color = col)
end
lines!(a2, [0, 1], [0, 1]; color = :black, linestyle = :dot)
axislegend(a1; position = :rb, labelsize = 11)
savefig("exo101326_test.png", fig)
fig

# %% [markdown]
# **Reading section 12: the test set confirms the cross-validation results.**
# - **Headline, DR25 KOIs:**
#
#   | Model | Test ROC-AUC [95% CI] | CV expected |
#   |---|---|---|
#   | Catalog-only | **0.957** [0.944, 0.970] | 0.954 |
#   | Catalog + centroid | **0.988** [0.982, 0.994] | 0.983 |
#   | Robovetter `koi_score` (reference) | 0.979 | — |
#
#   Every CV estimate lies inside the test CI, for both models and every subset. The test split is slightly easier
#   than the pool (all subsets score a little higher), which is within its sampling noise.
# - **The 98% planet-recall operating point holds:**
#   - Catalog-only (t = 0.145): recall 0.988 [0.975, 0.997], precision 0.77, keeps 31% of FPs (CV: 35%).
#   - Catalog + centroid (t = 0.211): recall 0.993 [0.983, 1.000], precision 0.87, keeps 16% of FPs (CV: 17%).
#   - No CONFIRMED planet in the DR25 test KOIs is lost at this threshold, by either model.
# - **Best F1 (for comparison with the literature):** catalog-only recall 0.90 / precision 0.89; catalog + centroid
#   0.96 / 0.95.
# - **Supplemental KOIs (117):** 0.72 and 0.79, also in line with CV (wide CIs, as expected).
# - **The numbers are final.** The test set has been used; any later model change has to be judged by CV again and
#   reported as post-test.

# %% [markdown]
# ## 13. Post-test experiment: train on CONFIRMED vs FP only, score CANDIDATEs
# - **Why:** CANDIDATE is the noisiest label. In step 2, many "missed planets" were CANDIDATEs that look like FPs.
#   Training only on CONFIRMED vs FALSE POSITIVE (6,598 of 8,288 pool KOIs) gives the model cleaner labels.
#   CANDIDATEs are then scored as an application.
# - **Setup:** the same final recipes (catalog-only and catalog + centroid, MLP + trees average), the same folds,
#   5 × 10. CANDIDATEs are dropped from *training* only, so every KOI still gets an OOF score and the runs pair with
#   the train-on-all references. Compute: `sweeps/labels_confirmed.jl`.
# - ⚠ **This comes after the sealed test.** It is judged by CV only and does not change the frozen models or the test
#   numbers.

# %%
cand = pool.koi_disposition .== "CANDIDATE"
mq_cf, tr_cf = load_cv("cv_mlpq_conftrain_r5s10.jld2"), load_cv("cv_evotree_conftrain_r5s10.jld2")
mq_cfc, tr_cfc = load_cv("cv_mlpq_centroid_conftrain_r5s10.jld2"), load_cv("cv_evotree_centroid_conftrain_r5s10.jld2")
avg_cf, avg_cfc = avg_of(mq_cf, tr_cf), avg_of(mq_cfc, tr_cfc)
cf_rows = NamedTuple[]
for (mname, A, B) in (("catalog-only", avg_cf, avg4), ("catalog + centroid", avg_cfc, avg_c))
    for (sname, m) in (("DR25 CONFIRMED vs FP", dr .& .!cand), ("DR25, CANDIDATE = planet", dr),
                       ("supplemental CONFIRMED vs FP", .!dr .& .!cand))
        c = cmp_on(A, B, m)
        push!(cf_rows, (model = mname, scored_on = sname, n = sum(m), train_all = round(c.B; digits = 4),
                        train_confirmed_only = round(c.A; digits = 4),
                        delta = @sprintf("%+.4f [%+.4f, %+.4f]", c.delta, c.ci_lo, c.ci_hi)))
    end
end
DataFrame(cf_rows)

# %% [markdown]
# ### 13.1 What the CANDIDATEs look like to each model
# Thresholds: each model's own 98%-planet-recall cut, set on DR25 **CONFIRMED** planets (pooled OOF, descriptive).
# "Look like FPs" = CANDIDATEs below that cut.

# %%
cand_rows = NamedTuple[]
cand_scores = Dict{String, Vector{Float64}}()
for (mname, cv) in (("catalog-only, train all", avg4), ("catalog-only, CONFIRMED-only", avg_cf),
                    ("+ centroid, train all", avg_c), ("+ centroid, CONFIRMED-only", avg_cfc))
    s = oof_scores(cv); cand_scores[mname] = s
    t98 = pick_threshold(yf[dr .& .!cand], s[dr .& .!cand], (:recall, 0.98))
    m = cand .& dr; mk2 = m .& isfinite.(ks)
    push!(cand_rows, (model = mname, t98 = round(t98; digits = 3), dr25_candidates = sum(m),
                      mean_score = round(mean(s[m]); digits = 3), below_0_5 = round(mean(s[m] .< 0.5); digits = 3),
                      look_like_fps = round(mean(s[m] .< t98); digits = 3),
                      fp_kept_at_t98 = round(mean(s[dr .& (yf .== 0)] .>= t98); digits = 3),
                      spearman_vs_koi_score = round(corspearman(s[mk2], ks[mk2]); digits = 3)))
end
DataFrame(cand_rows)

# %%
fig = Figure(size = (900, 400))
for (j, (ttl, k1, k2)) in enumerate((("catalog-only", "catalog-only, train all", "catalog-only, CONFIRMED-only"),
                                     ("catalog + centroid", "+ centroid, train all", "+ centroid, CONFIRMED-only")))
    local ax = Axis(fig[1, j]; xlabel = "planet score", ylabel = j == 1 ? "DR25 CANDIDATEs" : "", title = "CANDIDATE scores, $ttl")
    hist!(ax, cand_scores[k1][cand .& dr]; bins = 0:0.05:1, color = (:gray, 0.5), label = "train on all")
    hist!(ax, cand_scores[k2][cand .& dr]; bins = 0:0.05:1, color = (:darkorange, 0.45), label = "CONFIRMED-only")
    j == 1 && axislegend(ax; position = :lt)
end
savefig("exo101326_confirmed_only.png", fig)
fig

# %% [markdown]
# **Reading section 13:**
# - **Cleaner labels give a cleaner CONFIRMED-vs-FP separation.** DR25 CONFIRMED vs FP: catalog-only 0.978 → **0.982**
#   (+0.004 [+0.003, +0.006]); catalog + centroid 0.9945 → **0.9958** (+0.0013 [+0.0005, +0.0020]). Both are real.
# - **But it's worse at the actual task**, where CANDIDATEs count as planets: DR25 0.955 → 0.941 (catalog-only) and
#   0.983 → 0.976 (centroid). Supplemental KOIs: no clear change (CIs include 0).
# - **The CONFIRMED-only models are much more sceptical of CANDIDATEs.**
#   - DR25 CANDIDATEs' mean score falls from 0.69 to 0.41 (catalog-only) and from 0.78 to 0.49 (centroid).
#   - 36–41% of DR25 CANDIDATEs fall below the cut that keeps 98% of CONFIRMED planets (train-all models: 15–19%).
#   - Their ranking of CANDIDATEs agrees better with the robovetter's `koi_score` (Spearman 0.24 → 0.32; 0.33 → 0.38).
# - **Caveat: "CONFIRMED" is a biased sample.** Confirmed planets are the *confirmable* ones: higher SNR, often in
#   multi-planet systems (statistical validation). A CONFIRMED-only model partly learns "looks confirmable", so a low
#   score for a CANDIDATE means "unlike the confirmed planets", not necessarily "false positive".
# - **Decision-ready conclusion:** keep the frozen train-on-all models as the headline (they are what the test
#   validated, and they do the stated task better). The CONFIRMED-only model is a possible *secondary* score for
#   prioritising CANDIDATE follow-up, not a replacement.
