# Closing the MLP-vs-trees gap (follow-up to step 3).
#   screen: MLP variants on the TUNING folds (fold_seed0 = 5000, 2 shuffles x 3 seeds), the same folds
#           the tree grid used (MLP baseline there: 0.9255; best tree: 0.9318)
#   final:  chosen variants on the STANDARD folds (5 shuffles x 10 seeds), comparable with
#           cv_baseline_r5s10.jld2 and cv_evotree_tuned_r5s10.jld2
# Resumable: every finished run is saved and skipped next time.
#   julia -t auto sweeps/cv_mlp_gap.jl screen [names...]
#   julia -t auto sweeps/cv_mlp_gap.jl final name1 [name2 ...]
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

# AdamW decay is lr-coupled (decay per step = lr * wd), so wd ~ 1e-2..1e-1 is the useful range.
const VARIANTS = Dict(
    "base"          => (;),
    "epochs600"     => (epochs = 600, patience = 50),
    "wide128"       => (hidden = (128, 64),),
    "wide256"       => (hidden = (256, 128),),
    "deep3"         => (hidden = (128, 64, 32),),
    "drop0.1"       => (dropout = 0.1,),
    "drop0.3"       => (dropout = 0.3,),
    "lr3e-3"        => (lr = 3e-3,),
    "wd0.01"        => (wd = 0.01,),
    "wd0.1"         => (wd = 0.1,),
    "batch64"       => (batchsize = 64,),
    # combinations of the screen's winners (lr 3e-3, 600-epoch cap, more capacity)
    "lr3_e600"        => (lr = 3e-3, epochs = 600, patience = 50),
    "lr3_e600_wide"   => (lr = 3e-3, epochs = 600, patience = 50, hidden = (256, 128)),
    "lr3_e600_deep3"  => (lr = 3e-3, epochs = 600, patience = 50, hidden = (128, 64, 32)),
    # step 4: input transforms on the tuned MLP (lr3_e600 is the new MLP reference)
    "lr3_e600_logmore"  => (lr = 3e-3, epochs = 600, patience = 50, transform = :log_more),
    "lr3_e600_quantile" => (lr = 3e-3, epochs = 600, patience = 50, transform = :quantile),
)
const SCREEN_ORDER = ["base", "epochs600", "wide128", "wide256", "deep3", "drop0.1", "drop0.3",
                      "lr3e-3", "wd0.01", "wd0.1", "batch64"]

df = clean_koi(load_koi())
pool, _ = cv_pool(df)
y = pool.label
const SCREEN_DIR = joinpath(CV_DIR, "mlp_screen")
mkpath(SCREEN_DIR)

mode = isempty(ARGS) ? "screen" : ARGS[1]
names_ = length(ARGS) > 1 ? ARGS[2:end] : SCREEN_ORDER

if mode == "screen"
    for name in names_
        file = joinpath("mlp_screen", "screen_$(name).jld2")
        isfile(joinpath(CV_DIR, file)) && (println("skip (done): ", name); continue)
        t0 = time()
        cv = cv_run(pool; repeats = 2, seeds = 1:3, fold_seed0 = 5000, name, verbose = false, VARIANTS[name]...)
        save_cv(file, cv)
        s = oof_scores(cv)
        @printf("screen %-12s AUC %.4f  logloss %.4f  median best epoch %5.0f  max %4d  (%.1f min)\n", name,
                fast_auc(y, s), logloss(y, s), median(cv.results.best_epoch), maximum(cv.results.best_epoch), (time() - t0) / 60)
        flush(stdout)
    end
    # summary, paired against the screen baseline
    base = load_cv(joinpath("mlp_screen", "screen_base.jld2"))
    rows = NamedTuple[]
    for name in SCREEN_ORDER ∪ names_
        f = joinpath("mlp_screen", "screen_$(name).jld2")
        isfile(joinpath(CV_DIR, f)) || continue
        cv = load_cv(f)
        c = compare_cv(cv, base, pool; B = 1000)
        push!(rows, (variant = name, roc_auc = c.A, d_auc = c.delta, ci_lo = c.ci_lo, ci_hi = c.ci_hi,
                     logloss = logloss(y, oof_scores(cv)), median_epoch = median(cv.results.best_epoch),
                     at_cap = mean(cv.results.best_epoch .>= get(VARIANTS[name], :epochs, 200) - get(VARIANTS[name], :patience, 25))))
    end
    tbl = sort!(DataFrame(rows), :roc_auc; rev = true)
    CSV.write(joinpath(CV_DIR, "mlp_screen.csv"), tbl)
    show(tbl; allrows = true); println()
elseif mode == "final"
    for name in names_
        file = "cv_mlp_$(name)_r5s10.jld2"
        isfile(joinpath(CV_DIR, file)) && (println("skip (done): ", name); continue)
        cv = cv_run(pool; repeats = 5, seeds = 1:10, name, VARIANTS[name]...)
        println("saved: ", save_cv(file, cv))
    end
    base = load_cv("cv_baseline_r5s10.jld2")
    tree = load_cv("cv_evotree_tuned_r5s10.jld2")
    null = null_differences(base, pool)
    for name in names_
        cv = load_cv("cv_mlp_$(name)_r5s10.jld2")
        c = compare_cv(cv, base, pool; null)
        t = compare_cv(cv, tree, pool)
        @printf("[%s] AUC %.4f | vs MLP baseline Δ %+.4f [%+.4f, %+.4f] p_null %.3f | vs tuned trees Δ %+.4f [%+.4f, %+.4f]\n",
                name, c.A, c.delta, c.ci_lo, c.ci_hi, c.p_null, t.delta, t.ci_lo, t.ci_hi)
    end
end
println("DONE")
