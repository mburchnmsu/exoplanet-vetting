# %% [markdown]
# # Kepler vetting: noise floor and error analysis (10/10/2026)
#
# **Step 1, noise floor:** how much do the baseline MLP's scores move from luck alone (model seed, which stars land
# in validation, fold choice)? That sets the bar any later change (steps 3–5) has to clear.
#
# **Step 2, error analysis:** using an out-of-fold score for every KOI in the pool, where does the model fail and what
# kinds of false positives does it miss?
#
# Setup: baseline from `exo100926` (11 features, 64-32 MLP, dropout 0.2, Adam 1e-3, batch 128, early stopping with
# patience 25). Heavy compute ran headless in `sweeps/cv_noise_floor.jl`; this notebook only loads and analyzes it.
# **The test split (split seed 42) is never used here.**
#
# ## Key results
# - **Baseline:** pooled out-of-fold ROC-AUC **0.925** (95% star-bootstrap CI 0.919–0.931); **0.964** on CONFIRMED vs FP.
#   exo100926's 0.928 was a typical draw, not a lucky one.
# - **A single notebook run can't rank models.** On a ~930-KOI validation set, *which stars land in validation* moves
#   ROC-AUC by ±0.007 (range 0.914–0.943), five times more than the model seed (±0.0016). Two single runs need to differ by
#   ≳0.02 before it means anything.
# - **With repeated CV it can.** On identical folds, two runs of the same config differ by < 0.0005 (95%). From here on,
#   compare configs with `cv_run` + `compare_cv`, and require the paired star-bootstrap CI to exclude 0 as well.
# - **Seed ensembling** adds only +0.0017 AUC and saturates by ~5 seeds; use 5 seeds for comparisons.
# - **The sealed test set (~910 KOIs) has ±0.009 AUC sampling error on its own**, so use it once, at the end, as a check rather than to choose between models.
# - **Errors are concentrated, not uniform:**
#   1. *Supplemental KOIs* (not re-detected by the final DR25 pipeline, no `koi_score`): 13% of KOIs, a third of all
#      errors, AUC **0.65**, vs **0.948** for DR25 KOIs. The 570 found only by the old Q1–Q16 run score AUC 0.56.
#   2. *Eclipsing binaries are largely solved*: only 4.5% of pure stellar-eclipse (SS) FPs get through; odd/even-depth and
#      V-shaped FPs are caught ≥ 97% of the time, secondary-eclipse FPs 91–93%.
#   3. *Centroid-offset / contamination FPs are the main blind spot* (38% of CO-only FPs missed; CO/EC flags on 58% of all
#      missed FPs). Nothing in the 11 catalog features measures where on the sky the signal comes from.
#   4. *The multiplicity prior is very strong*: FPs in systems with ≥ 3 KOIs are missed 74–100% of the time, while 94% of
#      missed planets are in single-KOI systems.
#   5. *Most "missed planets" are CANDIDATEs* (82%); CONFIRMED planets are recovered 95% of the time. Many of the lowest-scored
#      CANDIDATEs are physically implausible (R > 25 R⊕, grazing) and likely label noise; the rest are small
#      ultra-short-period signals, the same blind spot that costs the model some confirmed USP planets (70% of planets
#      with P < 1.5 d are called FPs).
# - **Robustness (section 3):** all of these hold across 5 independent fold shuffles. Findings 2–4 also hold on DR25
#   KOIs alone, where the baseline reaches AUC 0.948 (CI 0.943–0.952). The long-period and low-SNR weak spots are
#   mostly a supplemental-KOI effect.

# %%
include(joinpath(@__DIR__, "src", "VettingSetup.jl"))
df_raw = load_koi()
df = clean_koi(df_raw)
pool, _ = cv_pool(df)                        # train+val of the main split; test stays sealed
cv = load_cv("cv_baseline_r5s10.jld2")
@assert cv.kepoi == pool.kepoi_name
single = JLD2.load(joinpath(CV_DIR, "single_split_runs.jld2"), "single")
R, S, K = size(cv.oof, 2), size(cv.oof, 3), maximum(cv.folds)
println("pool: $(nrow(pool)) KOIs on $(length(unique(pool.kepid))) stars")
println("CV: $K folds × $R fold shuffles × $S model seeds = $(nrow(cv.results)) trained models")

# %% [markdown]
# ## 1. Noise floor
# ### 1.1 Single models on held-out folds (~1,660 KOIs each)
# Each row summarizes all 250 models. `*_conf` = CONFIRMED vs FALSE POSITIVE only (CANDIDATEs dropped).

# %%
metric_cols = [:roc_auc, :pr_auc, :accuracy, :f1, :logloss, :roc_auc_conf, :pr_auc_conf, :best_epoch]
DataFrame(metric = string.(metric_cols),
          mean = [mean(cv.results[!, c]) for c in metric_cols],
          std  = [std(cv.results[!, c]) for c in metric_cols],
          min  = [minimum(cv.results[!, c]) for c in metric_cols],
          max  = [maximum(cv.results[!, c]) for c in metric_cols])

# %% [markdown]
# ### 1.2 Where the spread comes from (ROC-AUC)

# %%
ns = noise_summary(cv, pool)
pair_diffs = [ns.single[r, f, s1] - ns.single[r, f, s2] for r in 1:R, f in 1:K, s1 in 1:S, s2 in 1:S if s1 < s2]
nulls = null_differences(cv, pool; n = 500)
noise_tbl = DataFrame(
    source = ["single model, one fold (std over all 250)",
              "model seed only (std across seeds, fold fixed)",
              "two single models on the same fold (std of difference)",
              "fold-to-fold (10-seed ensemble per fold)",
              "pooled OOF, single seed (std over repeats × seeds)",
              "pooled OOF, 10-seed ensemble (std over 5 fold shuffles)",
              "null: 5-seed vs 5-seed ensembles, all repeats (std)",
              "null: 95th percentile of |difference|"],
    roc_auc = [std(ns.single), ns.seed_std_within_fold, std(pair_diffs), ns.fold_std_ens,
               std(ns.pooled_single), std(ns.pooled_ens), std(nulls), quantile(abs.(nulls), 0.95)])
println(@sprintf("final pooled OOF ROC-AUC (50 models per KOI): %.4f", ns.final))
println(@sprintf("mean single-model fold AUC %.4f -> 10-seed ensemble %.4f  (gain %+.4f)",
                 mean(ns.single), mean(ns.ens), ns.ensemble_gain))
noise_tbl

# %% [markdown]
# **Reading this:** a single model scored on one fold varies by ±0.005, and almost all of that comes from *which KOIs are in
# the fold* (fold-to-fold ±0.0046), not from training randomness (seed ±0.0014). Once predictions are pooled over all
# ~8,300 KOIs, training noise nearly disappears (±0.0009 for one seed, ±0.0006 for a 10-seed ensemble), and two
# 5-seed runs of the same config agree to within 0.0005 (95%).
#
# ### 1.3 Notebook-style runs: how lucky was exo100926?
# 20 model seeds on the exact exo100926 split, and 20 different train/val re-splits of the pool (same sizes, model
# seed 1). Each point is one validation ROC-AUC on ~930 KOIs, i.e. exactly what one notebook run reports.

# %%
println(combine(groupby(single, :kind), nrow => :runs,
                :roc_auc => mean => :auc_mean, :roc_auc => std => :auc_std,
                :roc_auc => minimum => :auc_min, :roc_auc => maximum => :auc_max,
                :roc_auc_conf => mean => :conf_mean, :roc_auc_conf => std => :conf_std))
fig = Figure(size = (800, 380))
ax = Axis(fig[1, 1]; xlabel = "validation ROC-AUC (~930 KOIs)", yticks = (1:2, ["20 model seeds\n(split 42)", "20 re-splits\n(model seed 1)"]),
          title = "One notebook run = one point")
for (i, kind) in enumerate(unique(single.kind))
    x = single.roc_auc[single.kind .== kind]
    scatter!(ax, x, i .+ 0.12 .* randn(Xoshiro(i), length(x)); markersize = 10, color = (:steelblue, 0.7))
end
vlines!(ax, [0.928179]; color = :firebrick, linestyle = :dash, label = "exo100926 (0.928)")
vlines!(ax, [ns.final]; color = :black, label = @sprintf("pooled OOF, 50-model ensemble (%.3f)", ns.final))
axislegend(ax; position = :lt)
savefig("exo101026_single_runs.png", fig)
fig

# %% [markdown]
# **Reading this:** with the split fixed, 20 seeds span 0.923–0.930 (exo100926's 0.928 is inside that range). Re-drawing the
# validation stars spans 0.914–0.943. So a notebook comparison like "0.928 → 0.935" is well within luck; comparisons
# need the CV protocol below.
#
# ### 1.4 Ensembling and the null distribution
# Left: pooled OOF ROC-AUC vs number of seeds averaged (mean over 5 fold shuffles and 20 random seed subsets).
# Right: the gap between two runs of the **same** config (5 seeds each, identical folds). A real improvement in
# steps 3–5 has to land clearly outside this.

# %%
ens_curve = map(1:S) do m
    rng = Xoshiro(m)
    mean(fast_auc(pool.label, vec(mean(cv.oof[:, r, randperm(rng, S)[1:m]]; dims = 2))) for r in 1:R, _ in 1:20)
end
fig = Figure(size = (900, 380))
a1 = Axis(fig[1, 1]; xlabel = "seeds in ensemble", ylabel = "pooled OOF ROC-AUC", title = "Ensemble size")
scatterlines!(a1, 1:S, ens_curve)
a2 = Axis(fig[1, 2]; xlabel = "ΔAUC between identical configs", ylabel = "count", title = "Null distribution (500 seed splits)")
hist!(a2, nulls; bins = 30, color = (:gray, 0.6))
vlines!(a2, quantile(nulls, [0.025, 0.975]); color = :firebrick, linestyle = :dash)
savefig("exo101026_ensemble_null.png", fig)
println("ensemble curve: ", round.(ens_curve; digits = 4))
fig

# %% [markdown]
# ### 1.4b The comparison protocol for steps 3–5
# `compare_cv(cvA, cvB, pool; null = nulls)` checks a new config against the baseline with two tests:
# 1. **Beyond training noise:** |ΔAUC| must sit outside the null distribution above (`p_null` small).
# 2. **Holds for other stars:** the paired star-bootstrap 95% CI on ΔAUC must exclude 0.
#
# Sanity check: comparing seeds 1–5 against seeds 6–10 of the *same* baseline should pass neither test.

# %%
cvA = merge(cv, (oof = cv.oof[:, :, 1:5],))
cvB = merge(cv, (oof = cv.oof[:, :, 6:10],))
compare_cv(cvA, cvB, pool; null = nulls)

# %% [markdown]
# ### 1.5 Statistical uncertainty: what the sealed test set will be able to tell us
# The noise floor above is about *training* randomness. Separately, any finite evaluation set has sampling error.
# Star-bootstrap of the pooled OOF score (all ~8,300 KOIs), and the spread of AUC on random star subsets the size of
# the test split (~790 stars, ~910 KOIs).

# %%
s_final = oof_scores(cv)
bs = star_bootstrap(pool.label, s_final, pool.kepid; B = 2000)
stars_u = unique(pool.kepid)
sub_auc = map(1:1000) do b
    pick = Set(rand(Xoshiro(b), stars_u, 792))
    m = in.(pool.kepid, Ref(pick))
    fast_auc(pool.label[m], s_final[m])
end
println(@sprintf("pooled OOF ROC-AUC %.4f, 95%% star-bootstrap CI [%.4f, %.4f]", ns.final, bs.lo, bs.hi))
println(@sprintf("AUC on a test-sized subset: std %.4f, central 95%% range [%.4f, %.4f]",
                 std(sub_auc), quantile(sub_auc, 0.025), quantile(sub_auc, 0.975)))

# %% [markdown]
# **Reading this:** the pooled score is known to about ±0.006 (95%) as an estimate for new stars. A test-sized set alone
# carries ±0.009 (1σ) sampling error: fine for a final sanity check, useless for picking between models that differ by 0.005.
#
# ### 1.6 Training stability
# Best epoch chosen by early stopping, and how much each KOI's score moves between models.

# %%
score_sd = vec(std(reshape(cv.oof, nrow(pool), :); dims = 2))     # spread over all 50 models per KOI
fig = Figure(size = (900, 360))
a1 = Axis(fig[1, 1]; xlabel = "best epoch", ylabel = "models", title = "Early-stopping epoch (250 models)")
hist!(a1, cv.results.best_epoch; bins = 25, color = (:steelblue, 0.6))
a2 = Axis(fig[1, 2]; xlabel = "OOF score (mean of 50 models)", ylabel = "std across models", title = "Per-KOI score stability")
scatter!(a2, s_final, score_sd; markersize = 3, color = (:black, 0.25))
savefig("exo101026_stability.png", fig)
println(@sprintf("per-KOI score std: median %.3f, 90th pct %.3f; KOIs whose 0.5 call flips between models: %.1f%%",
                 median(score_sd), quantile(score_sd, 0.9),
                 100 * mean([length(unique(vec(cv.oof[i, :, :]) .>= 0.5)) > 1 for i in 1:nrow(pool)])))
fig

# %% [markdown]
# **Reading this:** early stopping lands anywhere from epoch ~30 to 200 (median ~100; a few runs hit the 200 cap, so
# the cap could be raised a little). Individual KOI scores wobble by ~0.03 between models, more near 0.5, so 17% of KOIs
# flip their hard call in at least one of the 50 models. Use the averaged score for any per-KOI statement.
#
# ## 2. Error analysis (out-of-fold scores)
# Every pool KOI gets the mean score of 50 models (10 seeds × 5 fold shuffles) that never saw its star. Robovetter
# flags, comments and `koi_score` are joined in **for analysis only**.

# %%
a = with_vetting_info(pool, df_raw, s_final)
println(evaluate_subsets(a, a.score))
fig = plot_confusion(a.label, a.score; title = "Pool OOF")
savefig("exo101026_oof_confusion.png", fig)
fig

# %% [markdown]
# ### 2.0b Error budget: where the mistakes at threshold 0.5 come from

# %%
fp_miss = a[(a.label .== 0) .& (a.score .>= 0.5), :]
pl_miss = a[(a.label .== 1) .& (a.score .< 0.5), :]
pct(x, n) = @sprintf("%4d  (%4.1f%%)", x, 100x / n)
nf, np = nrow(fp_miss), nrow(pl_miss)
println("FALSE POSITIVES called planets: $nf")
println("  in multi-KOI systems (koi_count ≥ 2)       ", pct(sum(fp_miss.koi_count .>= 2), nf))
println("  flagged CO and/or EC (centroid/contam.)    ", pct(sum((fp_miss.koi_fpflag_co .== 1) .| (fp_miss.koi_fpflag_ec .== 1)), nf))
println("  flagged NT (not transit-like)              ", pct(sum(fp_miss.koi_fpflag_nt .== 1), nf))
println("  flagged SS (stellar eclipse)               ", pct(sum(fp_miss.koi_fpflag_ss .== 1), nf))
println("  model SNR < 15                             ", pct(sum(fp_miss.koi_model_snr .< 15), nf))
println("\nPLANETS called false positives: $np")
println("  CANDIDATE                                  ", pct(sum(pl_miss.koi_disposition .== "CANDIDATE"), np))
println("  CONFIRMED                                  ", pct(sum(pl_miss.koi_disposition .== "CONFIRMED"), np))
println("  period < 1.5 d                             ", pct(sum(pl_miss.koi_period .< 1.5), np))
println("  prad > 15 R⊕                               ", pct(sum(pl_miss.koi_prad .> 15), np))
println("  impact > 0.9 (near-grazing)                ", pct(sum(pl_miss.koi_impact .> 0.9), np))
println("  model SNR < 15                             ", pct(sum(pl_miss.koi_model_snr .< 15), np))
println("  koi_count = 1                              ", pct(sum(pl_miss.koi_count .== 1), np))

# %%
# KOIs with no robovetter koi_score: who are they, and how does the model do on them?
nos = ismissing.(a.koi_score)
println(@sprintf("no koi_score: %d KOIs, AUC %.4f   |   with koi_score: %d KOIs, AUC %.4f",
                 sum(nos), fast_auc(a.label[nos], a.score[nos]), sum(.!nos), fast_auc(a.label[.!nos], a.score[.!nos])))
println("dispositions without koi_score: ", countmap(a.koi_disposition[nos]))
println("koi_pdisposition without koi_score: ", countmap(a.koi_pdisposition[nos]))
println(@sprintf("median model SNR  no score %.1f vs with score %.1f; median prad %.2f vs %.2f",
                 median(a.koi_model_snr[nos]), median(a.koi_model_snr[.!nos]), median(a.koi_prad[nos]), median(a.koi_prad[.!nos])))

# %% [markdown]
# **Why no `koi_score`?** All of them are *supplemental* KOIs (`koi_disp_prov = q1_q17_dr25_sup_koi`) whose transit
# signal (TCE) came from an older pipeline run (Q1–Q16 or DR24), or from no TCE at all, and was **not re-detected by the
# final, uniform DR25 run**, so the DR25 robovetter never scored them. Their parameters come from older fits and they
# sit near the detection limit. Breakdown by TCE delivery:

# %%
a.tce_source = String.(coalesce.(a.koi_tce_delivname, "none (no TCE)"))
prov = combine(groupby(a, :tce_source), nrow => :n, :label => mean => :planet_frac,
               [:label, :score] => ((y, s) -> fast_auc(y, s)) => :roc_auc,
               [:label, :score] => ((y, s) -> mean((s .>= 0.5) .!= (y .== 1))) => :error_rate,
               [:label, :score] => ((y, s) -> sum((s .>= 0.5) .!= (y .== 1))) => :errors,
               :koi_model_snr => median => :median_snr)
prov.share_of_all_errors = prov.errors ./ sum(prov.errors)
sort!(prov, :n; rev = true)

# %% [markdown]
# **Reading this:** the DR25-detected KOIs are where the model works (AUC 0.948). The 570 KOIs that only the older
# Q1–Q16 run detected are essentially **random (AUC 0.56, 48% error rate)** and alone produce 21% of all errors; DR24-only
# and no-TCE KOIs are poor too (0.78, 0.74). Their catalog parameters come from older fits at SNR ≈ 8–11, so they
# probably carry less information. Whether to keep them in training,
# score them separately, or drop them is a scoping decision for later (see conclusions).

# %% [markdown]
# ### 2.1 Missed false positives by robovetter flag
# `missed_frac` = fraction of that group the model scores ≥ 0.5 (calls a planet). Flags overlap: "has X" groups
# double count, "only X+Y" groups are exact combinations.

# %%
fp_flag_table(a)

# %%
fig = plot_score_by_flag(a)
savefig("exo101026_fp_by_flag.png", fig)
fig

# %% [markdown]
# **Reading this:** stellar-eclipse FPs (SS) sit near 0: big, deep, V-shaped, grazing signals are easy from catalog
# parameters. The hard groups are **CO-only** (centroid offset: the signal comes from a nearby star; 38% missed),
# **NT-only** (not transit-like: systematics or variability; 25% missed, with a broad bump of scores around 0.3–0.5), and
# **EC-only** (ephemeris match: contamination from another known source; 23% missed). FPs flagged by several tests are
# easier, presumably because multiple problems leave multiple traces in the parameters.
#
# ### 2.2 Robovetter comment codes
# Codes carried by at least 40 FPs in the pool, sorted by how often the model misses them.

# %%
comment_token_table(a; min_n = 40)

# %% [markdown]
# **Reading this:** every eclipsing-binary signature is caught ≥ 91% of the time: odd/even depth (`*_ODDEVEN_*`),
# `DEEP_V_SHAPED`, `SWEET_EB` and `PLANET_IN_STAR` at ≥ 97%, secondary eclipses (`MOD_SEC_*`, `HAS_SEC_TCE`) at 91–93%. The most-missed codes need information the table
# does not have: aperture/pixel diagnostics (`CLEAR_APO`, `HALO_GHOST`, `CENT_*_OFFSET`), per-transit inspection
# (`INDIV_TRANS_SKYE`: individual transits on known bad "sky" columns), and alternative-detrending uniqueness tests
# (`MOD_NONUNIQ_ALT`). These are exactly what light curves and centroid time series (Phase 2) would provide.
#
# ### 2.3 Errors across parameter space
# Quantile bins of each raw feature. Red = FPs called planets, blue = planets called FPs, green = ROC-AUC within the
# bin (how well the model separates KOIs that look alike in that parameter).

# %%
fig = plot_binned_errors(a, [:koi_period, :koi_depth, :koi_prad, :koi_model_snr, :koi_impact, :koi_duration,
                             :koi_teq, :koi_steff, :koi_srad];
                         logx = [:koi_period, :koi_depth, :koi_prad, :koi_model_snr, :koi_srad, :koi_duration])
savefig("exo101026_binned_errors.png", fig)
fig

# %% [markdown]
# **Reading this:**
# - **SNR is the master variable**: within-bin AUC climbs from ~0.67 at SNR ≈ 7 to ~0.95 above SNR ≈ 30. Most errors
#   of both kinds are low-SNR KOIs (62% of missed FPs, 58% of missed planets have SNR < 15).
# - **Ultra-short periods (< 1.5 d)**: most real planets there are called FPs. FPs dominate that period range, so the
#   model learned "short period → FP".
# - **Long periods (> 100 d)**: few transits, within-bin AUC drops to ~0.63.
# - **Large radii / deep transits / impact > 1**: "planets" there are almost all called FPs. These are mostly CANDIDATEs
#   with R > 15 R⊕, which are physically more likely to be stars than planets (see 2.4).
# - **Stellar parameters** (Teff, R★) barely matter except at the hot / large-star end.

# %%
# Multiplicity (koi_count is discrete, so group directly)
combine(groupby(a, :koi_count), nrow => :n, :label => mean => :planet_frac,
        [:label, :score] => ((y, s) -> mean(s[y .== 0] .>= 0.5)) => :fp_miss,
        [:label, :score] => ((y, s) -> mean(s[y .== 1] .< 0.5)) => :planet_miss)

# %% [markdown]
# **Reading this:** multiplicity is a real astrophysical prior (Lissauer et al. 2012: multis are mostly real), but the
# model leans on it hard. Nearly every FP in a 3+ KOI system is called a planet, and almost all missed planets are
# singles. A `koi_count` ablation (step 4) will show how much of the AUC this one feature carries.
#
# ### 2.4 CONFIRMED vs CANDIDATE
# Both are labeled 1 in training. If CANDIDATE labels contain hidden false positives, CANDIDATE scores should sit lower
# and have a tail toward 0.

# %%
fig = plot_score_by_disposition(a)
savefig("exo101026_score_by_disposition.png", fig)
println(combine(groupby(a, :koi_disposition), nrow => :n, :score => median => :median,
                :score => (s -> quantile(s, 0.1)) => :p10, :score => (s -> mean(s .< 0.5)) => :frac_below_half))
fig

# %%
show_cols = [:kepoi_name, :koi_disposition, :score, :koi_score, :koi_period, :koi_depth, :koi_prad, :koi_impact,
             :koi_model_snr, :koi_count]
println("Lowest-scored CANDIDATEs (model thinks most FP-like):")
println(first(sort(a[a.koi_disposition .== "CANDIDATE", show_cols], :score), 15))
println("\nLowest-scored CONFIRMED planets (model's worst misses on confirmed planets):")
println(first(sort(a[a.koi_disposition .== "CONFIRMED", show_cols], :score), 15))

# %%
println("Highest-scored FALSE POSITIVEs (most planet-like FPs):")
fp_top = first(sort(a[a.label .== 0, :], :score; rev = true), 15)
println(fp_top[:, [:kepoi_name, :score, :flag_pattern, :koi_period, :koi_depth, :koi_prad, :koi_impact, :koi_model_snr, :koi_count]])
println()
for r in eachrow(fp_top[1:8, :])
    println(rpad(r.kepoi_name, 12), join(r.comment_tokens, ", "))
end

# %% [markdown]
# **Reading this:**
# - **Lowest CANDIDATEs** come in two kinds. Most look like stellar companions: radii of 25–700 R⊕, depths up to 35%,
#   impact > 1 (grazing), several with `koi_score = 0` so the robovetter doubts them too. Here the model acts as a
#   label-noise detector, and these "errors" are arguably correct. The others are small (0.2–1.5 R⊕) ultra-short-period
#   signals (P < 2 d): the USP blind spot again.
# - **Lowest CONFIRMED planets** fall into two groups: a few large/grazing ones (e.g. K01450.01, 59 R⊕, b = 1.23) and many
#   **ultra-short-period small planets** (P < 1.5 d, R ≈ 1–2.7 R⊕), a real population sitting where FPs dominate.
# - **Most planet-like FPs** are small (1–3.5 R⊕), moderate-SNR signals in **multi-KOI systems**, flagged for centroid
#   offsets, contamination or not-transit-like behaviour. They are indistinguishable from planets in catalog parameters.
#
# ### 2.5 Comparison with the robovetter's `koi_score`
# `koi_score` comes out of the same vetting that set many dispositions, so its AUC against the labels is partly
# circular. This is a sanity check on *where* the two disagree, not a benchmark.

# %%
ks_sum, b = koi_score_comparison(a)
println(ks_sum)
fig = plot_koi_score_comparison(b)
savefig("exo101026_vs_koi_score.png", fig)
fig

# %% [markdown]
# **Reading this:** on the 7,199 KOIs it scores, the robovetter reaches AUC 0.970 vs our 0.948. That gap is partly
# circular, since `koi_score` helped set those labels, but it is also the expected gain from pixel-level and light-curve
# tests. The two scores rank KOIs similarly (Spearman 0.75) and agree on 86% of hard calls. The bright left column in
# both panels: many KOIs got `koi_score ≈ 0` while our model is unsure or says planet. Among planets these are
# KOIs the DR25 robovetter scored near 0 but whose cumulative disposition is still CANDIDATE/CONFIRMED (presumably
# from other vetting or follow-up). Among FPs they are the centroid / not-transit-like cases from 2.1–2.2.

# %%
# Save every pool KOI's OOF score so later steps can track the same hard cases
oof_tbl = a[:, [:kepoi_name, :kepid, :koi_disposition, :label, :score, :flag_pattern, :koi_score]]
oof_tbl.score_sd = score_sd
CSV.write(joinpath(CV_DIR, "oof_baseline_r5s10.csv"), oof_tbl)
println("wrote ", joinpath(CV_DIR, "oof_baseline_r5s10.csv"))

# %%
# Disagreements by disposition: model says planet but robovetter says FP-like, and vice versa
b.model_planet = b.score .>= 0.5
b.rv_planet = b.koi_score .>= 0.5
combine(groupby(b, :koi_disposition), nrow => :n,
        [:model_planet, :rv_planet] => ((m, r) -> mean(m .& .!r)) => :model_yes_rv_no,
        [:model_planet, :rv_planet] => ((m, r) -> mean(.!m .& r)) => :model_no_rv_yes,
        [:model_planet, :rv_planet] => ((m, r) -> mean(m .== r)) => :agree)

# %% [markdown]
# ## 3. Robustness checks
# ### 3.1 Noise thresholds for other metrics
# The same null test as 1.4 (5-seed vs 5-seed ensembles, identical folds) and a star bootstrap, for each metric a later
# comparison might use. `null_95` is the training-noise bar; `boot_halfwidth` is the 95% sampling uncertainty of the
# metric itself for this pool. Log-loss: lower is better.

# %%
conf = a.koi_disposition .!= "CANDIDATE"
cv_conf = merge(cv, (oof = cv.oof[conf, :, :], folds = cv.folds[conf, :], kepoi = cv.kepoi[conf]))
pool_conf = pool[conf, :]
prauc(y, p) = au_prcurve(Int.(y), p)
metric_specs = [("ROC-AUC", fast_auc, cv, pool), ("PR-AUC", prauc, cv, pool), ("log-loss", logloss, cv, pool),
                ("ROC-AUC, CONFIRMED vs FP", fast_auc, cv_conf, pool_conf),
                ("PR-AUC, CONFIRMED vs FP", prauc, cv_conf, pool_conf)]
thr = map(metric_specs) do (name, m, c, p)
    nd = null_differences(c, p; n = 300, metric = m)
    bs_m = star_bootstrap(p.label, oof_scores(c), p.kepid; B = 500, metric = m)
    (metric = name, baseline = m(p.label, oof_scores(c)), null_std = std(nd), null_95 = quantile(abs.(nd), 0.95),
     boot_lo = bs_m.lo, boot_hi = bs_m.hi, boot_halfwidth = (bs_m.hi - bs_m.lo) / 2)
end |> DataFrame

# %% [markdown]
# ### 3.2 Are the error findings stable?
# Each of the 5 fold shuffles gives an *independent* set of OOF predictions (different folds, so different training
# sets per KOI). Recomputing the headline error numbers per shuffle shows whether they depend on one lucky partition.
# `wilson95` is the binomial 95% interval for the 50-model ensemble's rate (sampling uncertainty from finite group size).

# %%
function wilson(k, n; z = 1.96)
    p = k / n; d = 1 + z^2 / n
    c = (p + z^2 / (2n)) / d; h = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / d
    return (c - h, c + h)
end
fp = a.label .== 0
groups = [("FP missed: SS only",   fp .& (a.flag_pattern .== "SS")),
          ("FP missed: CO only",   fp .& (a.flag_pattern .== "CO")),
          ("FP missed: NT only",   fp .& (a.flag_pattern .== "NT")),
          ("FP missed: EC only",   fp .& (a.flag_pattern .== "EC")),
          ("FP missed: koi_count ≥ 3", fp .& (a.koi_count .>= 3)),
          ("FP missed: koi_count = 1", fp .& (a.koi_count .== 1)),
          ("CONFIRMED called FP",  a.koi_disposition .== "CONFIRMED"),
          ("CANDIDATE called FP",  a.koi_disposition .== "CANDIDATE"),
          ("planet called FP: P < 1.5 d", (a.label .== 1) .& (a.koi_period .< 1.5))]
err_rate(s, m, isfp) = isfp ? mean(s[m] .>= 0.5) : mean(s[m] .< 0.5)
stab = map(groups) do (name, m)
    isfp = startswith(name, "FP")
    per_rep = [err_rate(vec(mean(cv.oof[:, r, :]; dims = 2)), m, isfp) for r in 1:R]
    k = isfp ? sum(a.score[m] .>= 0.5) : sum(a.score[m] .< 0.5)
    lo, hi = wilson(k, sum(m))
    (group = name, n = sum(m), ensemble = k / sum(m), per_shuffle_min = minimum(per_rep),
     per_shuffle_max = maximum(per_rep), wilson95_lo = lo, wilson95_hi = hi)
end |> DataFrame
src_auc = [(src, [fast_auc(a.label[a.tce_source .== src], vec(mean(cv.oof[a.tce_source .== src, r, :]; dims = 2))) for r in 1:R])
           for src in ("q1_q17_dr25_tce", "q1_q16_tce")]
for (src, v) in src_auc
    println(rpad(src, 18), "AUC per fold shuffle: ", round.(v; digits = 3))
end
stab

# %% [markdown]
# ### 3.3 Do the findings survive on DR25 KOIs alone?
# Supplemental KOIs are hard *and* differ in composition, so they could distort the flag, multiplicity and parameter
# results. Repeat the key tables on the 7,199 DR25-detected KOIs only (same OOF scores, no retraining).

# %%
a25 = a[a.tce_source .== "q1_q17_dr25_tce", :]
bs25 = star_bootstrap(a25.label, a25.score, a25.kepid; B = 2000)
println(@sprintf("DR25 KOIs: n = %d, ROC-AUC %.4f (95%% CI %.4f–%.4f)", nrow(a25), fast_auc(a25.label, a25.score), bs25.lo, bs25.hi))
println(evaluate_subsets(a25, a25.score))
println()
println(first(fp_flag_table(a25), 12))
println()
println(combine(groupby(a25, :koi_count), nrow => :n, :label => mean => :planet_frac,
                [:label, :score] => ((y, s) -> mean(s[y .== 0] .>= 0.5)) => :fp_miss,
                [:label, :score] => ((y, s) -> mean(s[y .== 1] .< 0.5)) => :planet_miss))

# %%
fig = plot_binned_errors(a25, [:koi_period, :koi_prad, :koi_model_snr]; ncols = 3,
                         logx = [:koi_period, :koi_prad, :koi_model_snr])
Label(fig[-1, :], "DR25-detected KOIs only"; fontsize = 13)
savefig("exo101026_binned_errors_dr25.png", fig)
fig

# %% [markdown]
# **Reading 3.1–3.3:**
# - **Thresholds for later comparisons** (two runs of the same config, 95%): ROC-AUC 0.0005, PR-AUC 0.0006,
#   log-loss 0.0011, CONFIRMED-vs-FP ROC-AUC 0.0005, CONFIRMED-vs-FP PR-AUC 0.0011. The metrics themselves carry
#   ±0.004–0.012 sampling uncertainty for this pool, which is why paired comparisons are needed.
# - **The error findings are not partition luck.** Every headline rate moves by ≤ 0.03 across the 5 independent fold
#   shuffles. Q1–Q16 KOIs score AUC 0.54–0.56 in every shuffle; DR25 KOIs 0.946–0.947.
# - **Ultra-short periods are a real blind spot:** 70% of planets with P < 1.5 d (n = 201) are called FPs.
# - **On DR25 KOIs alone** (AUC 0.948, 95% CI 0.943–0.952; CONFIRMED vs FP 0.970):
#   - the centroid-offset blind spot persists (36% of CO-only FPs missed);
#   - the multiplicity effect persists (FPs with 3+ KOIs: 43–100% missed by count; planet misses almost all singles);
#   - EB-type FPs stay easy (4.5% missed);
#   - the "not transit-like" problem shrinks (NT-only missed 18% vs 25%), and the long-period and low-SNR weak spots
#     mostly disappear (within-bin AUC ≈ 0.86 at P > 100 d vs 0.63, ≈ 0.80 at the lowest SNR vs 0.67). Much of those
#     came from the supplemental KOIs.
#
# ## 4. Conclusions and what they mean for steps 3–5
#
# **Noise floor (step 1).** Always compare configs with star-grouped repeated CV on the same folds (`cv_run` with
# `repeats = 5`, `seeds = 1:5`, default `fold_seed0`; ~8 min per config), then `compare_cv` against
# `cv_baseline_r5s10.jld2`. Treat a change as real only if |ΔAUC| clears the training-noise null (95%: 0.0005) **and**
# the paired star-bootstrap CI excludes 0. Single notebook runs on the 80/10 split can't separate configs closer than ~0.02.
#
# **Error analysis (step 2).** The baseline's errors have three main causes:
# 1. **Supplemental (non-DR25) KOIs**: low-SNR legacy detections with older parameters. AUC 0.65 on 13% of the data;
#    a third of all errors.
# 2. **Missing information**: centroid offsets, contamination, per-transit artifacts. The 11 catalog features can't see
#    these; they need pixel/centroid data or light curves.
# 3. **Learned priors that are mostly right but sometimes wrong**: multiplicity (FPs in multis pass), short period → FP
#    (USP planets fail), huge radius → FP (correct, and it flags suspicious CANDIDATEs).
#
# **Suggested order for later:**
# - *Step 3 (baselines):* logistic regression and gradient-boosted trees (EvoTrees.jl) through the same CV protocol.
# - *Step 4 (features):*
#   - `koi_count` ablation, to see how much the multiplicity prior carries;
#   - log-transform `koi_model_snr`;
#   - `koi_insol` back in;
#   - decide whether to include centroid observables (`koi_dicco_*`, `koi_dikco_*`, `koi_fwm_*`), which target the
#     biggest blind spot but are robovetter inputs;
#   - decide how to treat supplemental KOIs: train/evaluate on DR25 only, report separately, or add a
#     "DR25-detected" indicator. A **scoping decision** that also affects how the results compare to published DR25
#     vetting work.
# - *Label choice:* train on CONFIRMED vs FP only (cleaner labels) and score CANDIDATEs as an application.
# - *Step 5 (operating point):* calibration and threshold choice on the pooled OOF scores (8,300 KOIs, not 930).
# - *Phase 2:* light curves and centroid time series target exactly the CO / NT / EC failures found here.
