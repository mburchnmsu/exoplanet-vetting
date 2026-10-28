# Phase 2, M4: CNNs on the light-curve views (src/CNN.jl), on the standard folds so every run is paired with
# Phase 1 / 2a. Variants (local-tower channels; tabular inputs joined before the dense head):
#   views     : local, odd, even, secondary;           no tabular features         (2b-1)
#   hybrid    : local, odd, even, secondary;           + FEATURES                  (2b-2)
#   hybrid_lc : local, odd, even, secondary;           + FEATURES + LC_FEATURES
#   cent      : + cen1, cen2 channels;                 + FEATURES + CENTROID_FEATURES (2b-3)
#   cent_lc   : + cen1, cen2 channels;                 + all of the above + LC_CENTROID_FEATURES
# Modes (resumable per variant):
#   --gate          time one epoch per variant + one full model on repeat 1 / fold 1 (plan §5 timing gate)
#   --screen [v..]  1 fold shuffle x 2 seeds (D6)             -> cv_cnn_<v>_r1s2.jld2
#   --final v [v..] 5 fold shuffles x 3 seeds (D6)            -> cv_cnn_<v>_r5s3.jld2
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p2_cnn.jl --gate"
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())
using Dates

df_raw = load_koi()
pool0, _ = cv_pool(clean_koi(df_raw))
pool = add_centroid_features(pool0, df_raw)
pool = leftjoin(pool, CSV.read(joinpath(CV_DIR, "p2_lcfeatures.csv"), DataFrame); on = :kepoi_name, order = :left)
@assert pool.kepoi_name == pool0.kepoi_name
for c in [LC_FEATURES; LC_CENTROID_FEATURES]
    pool[!, c] = Float64.(coalesce.(pool[!, c], NaN))
end
dr = is_dr25(pool, df_raw)
V = load_views()
@assert all(n -> haskey(V["__col"], n), pool.kepoi_name)

const FLUX4 = [:local_view, :odd, :even, :secondary]
const CEN6  = [FLUX4; :cen1; :cen2]
const VARIANTS = Dict(
    "views"     => (keys = FLUX4, features = FEATURES, use_feats = false),
    "hybrid"    => (keys = FLUX4, features = FEATURES, use_feats = true),
    "hybrid_lc" => (keys = FLUX4, features = [FEATURES; LC_FEATURES], use_feats = true),
    "cent"      => (keys = CEN6,  features = [FEATURES; CENTROID_FEATURES], use_feats = true),
    "cent_lc"   => (keys = CEN6,  features = [FEATURES; CENTROID_FEATURES; LC_FEATURES; LC_CENTROID_FEATURES], use_feats = true))
const ORDER = ["views", "hybrid", "hybrid_lc", "cent", "cent_lc"]
const CNN_KW = (epochs = 60, patience = 8, lr = 1e-3, wd = 1e-4, batchsize = 64)
stamp() = Dates.format(now(), "mm-dd HH:MM")

"Data of repeat 1 / fold 1, exactly as cv_run builds it."
function fold1(v)
    folds = star_folds(pool; k = 5, seed = 1000 + 1)
    held = folds .== 1
    tr, va = split_by_star(pool[.!held, :]; at = (1 - 1 / 9,), seed = 1000 + 100 + 1)
    te = pool[held, :]
    data = prepare_data(tr, va, te; features = v.features, transform = :quantile)
    return merge(data, view_extra(V, v.keys)(tr, va, te)), te
end

if "--gate" in ARGS
    println(stamp(), "  timing gate, ", Threads.nthreads(), " threads")
    for name in ORDER
        v = VARIANTS[name]
        data, _ = fold1(v)
        m, h = train_cnn(data; seed = 1, CNN_KW..., epochs = 2, patience = 99, use_feats = v.use_feats)  # epoch 1 compiles
        @printf("  %-10s params %7d   epoch %.1f s (first, incl. compile %.1f s)   RSS %.2f GB\n", name,
                n_params(m isa CNNNoFeats ? m.m : m), h.epoch_seconds[end], h.epoch_seconds[1], Sys.maxrss() / 1e9)
        flush(stdout)
    end
    name = "hybrid_lc"; v = VARIANTS[name]
    data, te = fold1(v)
    t0 = time()
    m, h = train_cnn(data; seed = 1, CNN_KW..., use_feats = v.use_feats, verbose = true)
    p = predict_test(m, data); ddr = is_dr25(te, df_raw)
    @printf("  full %s model, repeat 1 fold 1: %.1f min, best epoch %d of %d, fold AUC %.4f (DR25 %.4f)\n", name,
            (time() - t0) / 60, h.best_epoch, length(h.val), fast_auc(te.label, p), fast_auc(te.label[ddr], p[ddr]))
    # reference on the same fold: the Phase 2a trees + MLP average (seeds 1:2), from the saved CV
    a, b = load_cv("cv_mlpq_lc_r5s10.jld2"), load_cv("cv_evotree_lc_r5s10.jld2")
    hm = a.folds[:, 1] .== 1
    q = vec(mean((a.oof[hm, 1, 1:2] .+ b.oof[hm, 1, 1:2]) ./ 2; dims = 2))
    @printf("  same fold, Phase 2a MLP + trees (2 seeds): AUC %.4f (DR25 %.4f)\n", fast_auc(te.label, q),
            fast_auc(te.label[ddr], q[ddr]))
    println(stamp(), "  gate done")
    exit()
end

function run_cv(name, repeats, seeds, tag)
    file = "cv_cnn_$(name)_$(tag).jld2"
    isfile(joinpath(CV_DIR, file)) && (println("loaded: ", file); return load_cv(file))
    v = VARIANTS[name]
    t0 = time()
    println(stamp(), "  start ", file); flush(stdout)
    cv = cv_run(pool; repeats, seeds, name = "cnn_$name", features = v.features, transform = :quantile,
                trainer = train_cnn, extra = view_extra(V, v.keys), use_feats = v.use_feats, CNN_KW...)
    @printf("%s  saved %s (%.1f h); median best epoch %.0f\n", stamp(), save_cv(file, cv), (time() - t0) / 3600,
            median(cv.results.best_epoch))
    flush(stdout)
    return cv
end

"Restrict a CV result to rows `m`, repeats `r` and seeds `s` (to pair a screen with a longer run)."
sub(cv, m; r = Colon(), s = Colon()) = merge(cv, (oof = cv.oof[m, r, s], folds = cv.folds[m, r], kepoi = cv.kepoi[m]))
avg2(a, b) = merge(a, (oof = (a.oof .+ b.oof) ./ 2,))

"DR25 AUC of the 50/50 average with the matching Phase 2a model (paired, first R repeats): the screen's ranking."
function ens_auc(name, cv, R)
    ref = startswith(name, "cent") ? ("cv_mlpq_lccent_r5s10.jld2", "cv_evotree_lccent_r5s10.jld2") :
                                     ("cv_mlpq_lc_r5s10.jld2", "cv_evotree_lc_r5s10.jld2")
    base = sub(avg2(load_cv(ref[1]), load_cv(ref[2])), trues(nrow(pool)); r = 1:R)
    s = (oof_scores(base) .+ oof_scores(cv)) ./ 2
    return fast_auc(pool.label[dr], s[dr])
end

function report(name, cv, R, S)
    ref = startswith(name, "cent") ? ("cv_mlpq_lccent_r5s10.jld2", "cv_evotree_lccent_r5s10.jld2") :
                                     ("cv_mlpq_lc_r5s10.jld2", "cv_evotree_lc_r5s10.jld2")
    base = sub(avg2(load_cv(ref[1]), load_cv(ref[2])), trues(nrow(pool)); r = 1:R)    # all 10 seeds of 2a
    ens = merge(base, (oof = (mean(base.oof; dims = 3) .+ mean(cv.oof; dims = 3)) ./ 2,))
    for (lab, m) in [("DR25", dr), ("all", trues(nrow(pool)))]
        c = compare_cv(sub(cv, m), sub(base, m), pool[m, :]; B = 500)
        e = compare_cv(sub(ens, m), sub(base, m), pool[m, :]; B = 500)
        @printf("  %-10s %-4s CNN %.4f | Phase 2a %.4f: Δ %+.4f [%+.4f, %+.4f] | 50/50 with 2a %.4f: Δ %+.4f [%+.4f, %+.4f]\n",
                name, lab, c.A, c.B, c.delta, c.ci_lo, c.ci_hi, e.A, e.delta, e.ci_lo, e.ci_hi)
    end
    flush(stdout)
end

if "--screen" in ARGS
    names = filter(in(ORDER), ARGS); names = isempty(names) ? ORDER : names
    for name in names
        report(name, run_cv(name, 1, 1:2, "r1s2"), 1, 2)
    end
elseif "--final" in ARGS
    for name in filter(in(ORDER), ARGS)
        report(name, run_cv(name, 5, 1:3, "r5s3"), 5, 3)
    end
elseif "--auto" in ARGS
    # screen every variant, then the final 5 x 3 CV for the best one of each family (by DR25 AUC of its 50/50 with 2a)
    scr = Dict(name => run_cv(name, 1, 1:2, "r1s2") for name in ORDER)
    foreach(name -> report(name, scr[name], 1, 2), ORDER)
    for fam in (["views", "hybrid", "hybrid_lc"], ["cent", "cent_lc"])
        best = argmax(name -> ens_auc(name, scr[name], 1), fam)
        println(stamp(), "  family pick: ", best, "  (", join([@sprintf("%s %.4f", n, ens_auc(n, scr[n], 1)) for n in fam], ", "), ")")
        flush(stdout)
        report(best, run_cv(best, 5, 1:3, "r5s3"), 5, 3)
    end
end
println(stamp(), "  DONE")
