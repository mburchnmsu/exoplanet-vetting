# Step 3: baselines vs the MLP, on the SAME folds as cv_baseline_r5s10.jld2.
#   (A) logistic regression (Dense(11 => 1), same scaling / loss / early stopping as the MLP)
#   (B) gradient-boosted trees (EvoTrees) with default-ish settings
#   (C) small tree hyperparameter grid, scored on DIFFERENT fold shuffles (fold_seed0 = 5000)
#       so the choice isn't tuned on the folds used for the final comparison
#   (D) the chosen tree config on the standard folds
# All runs: 5 fold shuffles x 10 seeds, matching the MLP baseline. The test split is never used.
# Run headless:  julia -t auto sweeps/cv_baselines.jl
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
println("threads: ", Threads.nthreads())

df = clean_koi(load_koi())
pool, _ = cv_pool(df)
y = pool.label
base = load_cv("cv_baseline_r5s10.jld2")
null = null_differences(base, pool)

function report(name, cv)
    c = compare_cv(cv, base, pool; null)
    @printf("[%s] pooled OOF AUC %.4f   vs MLP %.4f   Δ %+.4f  CI [%+.4f, %+.4f]  p_null %.3f\n",
            name, c.A, c.B, c.delta, c.ci_lo, c.ci_hi, c.p_null)
    flush(stdout)
end

# Resumable: finished CV runs are loaded instead of re-run, and grid rows are
# checkpointed to evotree_grid_partial.csv after each config.
function cv_cached(file, name; kw...)
    isfile(joinpath(CV_DIR, file)) && (println("loaded: ", file); return load_cv(file))
    cv = cv_run(pool; repeats = 5, seeds = 1:10, name, kw...)
    println("saved: ", save_cv(file, cv))
    return cv
end

# --- (A) logistic regression -------------------------------------------------------
lr_cv = cv_cached("cv_logreg_r5s10.jld2", "logreg"; hidden = (), dropout = 0.0, lr = 1e-2)
report("logreg", lr_cv)

# --- (B) trees, defaults of train_evotree ------------------------------------------------
tree_cv = cv_cached("cv_evotree_default_r5s10.jld2", "evotree_default"; trainer = train_evotree)
report("evotree_default", tree_cv)

# --- (C) tuning grid on separate fold shuffles -----------------------------------------
grid = [(max_depth = d, min_weight = mw, L2 = l2, colsample = cs)
        for d in (3, 4, 5, 6, 8), mw in (1.0, 10.0, 40.0), l2 in (1.0, 10.0), cs in (0.5, 0.8)] |> vec
partial = joinpath(CV_DIR, "evotree_grid_partial.csv")
rows = NamedTuple[]
if isfile(partial)
    for r in eachrow(CSV.read(partial, DataFrame))
        push!(rows, (max_depth = r.max_depth, min_weight = Float64(r.min_weight), L2 = Float64(r.L2),
                     colsample = Float64(r.colsample), roc_auc = r.roc_auc, logloss = r.logloss,
                     median_rounds = r.median_rounds))
    end
    println("resuming grid: ", length(rows), " configs already done")
end
done(g) = any(r -> (r.max_depth, r.min_weight, r.L2, r.colsample) == (g.max_depth, g.min_weight, g.L2, g.colsample), rows)
for (i, g) in enumerate(grid)
    done(g) && continue
    t0 = time()
    cv = cv_run(pool; repeats = 2, seeds = 1:3, fold_seed0 = 5000, name = "grid", verbose = false,
                trainer = train_evotree, g...)
    s = oof_scores(cv)
    push!(rows, merge(g, (roc_auc = fast_auc(y, s), logloss = logloss(y, s),
                          median_rounds = median(cv.results.best_epoch))))
    @printf("grid %2d/%d %s  AUC %.4f  logloss %.4f  rounds %4d  (%.0fs)\n", i, length(grid), string(g),
            rows[end].roc_auc, rows[end].logloss, rows[end].median_rounds, time() - t0)
    CSV.write(partial, DataFrame(rows))
    flush(stdout)
end
gridres = sort!(DataFrame(rows), :roc_auc; rev = true)
CSV.write(joinpath(CV_DIR, "evotree_grid.csv"), gridres)
println(first(gridres, 10))

# MLP on the same tuning folds, as a reference point for the grid numbers
mlp_ref = cv_run(pool; repeats = 2, seeds = 1:3, fold_seed0 = 5000, name = "mlp_gridref", verbose = false)
@printf("MLP on tuning folds: AUC %.4f  logloss %.4f\n", fast_auc(y, oof_scores(mlp_ref)), logloss(y, oof_scores(mlp_ref)))
jldsave(joinpath(CV_DIR, "evotree_grid.jld2"); gridres, mlp_ref_auc = fast_auc(y, oof_scores(mlp_ref)),
        mlp_ref_logloss = logloss(y, oof_scores(mlp_ref)))

# --- (D) best grid config on the standard folds ------------------------------------------
best = gridres[1, :]
best_kw = (max_depth = best.max_depth, min_weight = best.min_weight, L2 = best.L2, colsample = best.colsample)
println("chosen tree config: ", best_kw)
tuned_cv = cv_cached("cv_evotree_tuned_r5s10.jld2", "evotree_tuned"; trainer = train_evotree, best_kw...)
report("evotree_tuned", tuned_cv)
println("DONE")
