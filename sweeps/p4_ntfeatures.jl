# Phase 4, M1 (N-a + N-b): uniqueness and per-transit consistency features (src/LCFeatures.jl, LC_NT_FEATURES) for
# every KOI on every cached star, one pass with the Phase 2 detrending (flux only).
# Resumable: shards of 250 stars in data/lightcurves/features_p4/shard_NNN.jld2; finished shards are skipped.
# Output: sweeps/results/p4_ntfeatures.csv (one row per KOI, no labels).
#   --classes  pool only: feature values per class and single-feature AUCs (labels for analysis, after the pass)
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p4_ntfeatures.jl"
#   ... sweeps/p4_ntfeatures.jl --test 40     time the first 40 stars, print a few rows, save nothing
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
using Dates

const FEAT_DIR = joinpath(LC_DIR, "features_p4")
const SHARD = 250
const OUT = joinpath(CV_DIR, "p4_ntfeatures.csv")

df_raw = load_koi()
stars = sort(unique(clean_koi(df_raw).kepid))
shard_path(i) = joinpath(FEAT_DIR, @sprintf("shard_%03d.jld2", i))
show_df(t) = (show(stdout, MIME"text/plain"(), t; allrows = true, allcols = true); println(); flush(stdout))

function feat_star(k)
    try
        return star_nt_features(k, df_raw), ""
    catch e
        return [], first(replace(sprint(showerror, e), r"[\r\n]" => " "), 200)
    end
end

function run_stars(ks)
    out = Vector{Any}(undef, length(ks))
    Threads.@threads :dynamic for i in eachindex(ks)
        out[i] = feat_star(ks[i])
    end
    return out
end

if "--test" in ARGS
    n = parse(Int, ARGS[findfirst(==("--test"), ARGS) + 1])
    run_stars(stars[1:2])                                     # compile
    t0 = time(); out = run_stars(stars[3:n+2]); dt = time() - t0
    rows = DataFrame(reduce(vcat, (first(o) for o in out); init = Any[]))
    @printf("test: %d stars, %d KOIs, %d errors, %.2f s/star wall with %d threads -> all %d stars ~%.1f h; RSS %.2f GB\n",
            n, nrow(rows), count(o -> !isempty(last(o)), out), dt / n, Threads.nthreads(),
            length(stars), dt / n * length(stars) / 3600, Sys.maxrss() / 1e9)
    foreach(o -> isempty(last(o)) || println("error: ", last(o)), out)
    show(stdout, MIME"text/plain"(), first(rows, 15); allcols = true, truncate = 0)
    println()
    exit()
end

if !("--classes" in ARGS)
    mkpath(FEAT_DIR)
    nsh = cld(length(stars), SHARD)
    t_start = time(); done_now = 0
    for i in 1:nsh
        isfile(shard_path(i)) && continue
        ks = stars[(i-1)*SHARD+1:min(i * SHARD, end)]
        out = run_stars(ks)
        fs = reduce(vcat, (first(o) for o in out); init = Any[])
        es = [(kepid = k, error = last(o)) for (k, o) in zip(ks, out) if !isempty(last(o))]
        jldsave(shard_path(i) * ".part"; feats = fs, errs = es)
        mv(shard_path(i) * ".part", shard_path(i); force = true)
        global done_now += 1
        el = (time() - t_start) / 3600
        left = count(j -> !isfile(shard_path(j)), 1:nsh)
        @printf("%s  shard %d/%d: %d KOIs, %d star errors; %.2f h elapsed, ~%.1f h left; RSS %.2f GB\n",
                Dates.format(now(), "HH:MM"), i, nsh, length(fs), length(es), el, el / done_now * left, Sys.maxrss() / 1e9)
        flush(stdout)
    end
    feats = Any[]; errs = Any[]
    for i in 1:nsh
        d = JLD2.load(shard_path(i))
        append!(feats, d["feats"]); append!(errs, d["errs"])
    end
    nt = DataFrame(feats)
    CSV.write(OUT, nt)
    isempty(errs) || CSV.write(joinpath(CV_DIR, "p4_ntfeatures_errors.csv"), DataFrame(errs))
    @printf("\nM1 features: %d KOIs on %d stars, %d star errors\n", nrow(nt), length(stars), length(errs))
    for c in LC_NT_FEATURES
        v = Float64.(coalesce.(nt[!, c], NaN)); ok = filter(isfinite, v)
        @printf("  %-12s finite %5d / %d   median %9.3g   [5%%, 95%%] %9.3g %9.3g\n", c, length(ok), length(v),
                median(ok), quantile(ok, 0.05), quantile(ok, 0.95))
    end
end

if "--classes" in ARGS          # labels on the CV pool only; the test split is never looked at by class
    pool, _ = cv_pool(clean_koi(df_raw))
    pool = leftjoin(pool, CSV.read(OUT, DataFrame); on = :kepoi_name, order = :left)
    a = with_vetting_info(pool, df_raw, zeros(nrow(pool)))
    pat = a.flag_pattern
    y = pool.label; dr = is_dr25(pool, df_raw)
    S = (repeat_scores(load_cv("cv_mlpq_eph_r5s10.jld2")) .+ repeat_scores(load_cv("cv_evotree_eph_r5s10.jld2"))) ./ 2
    s = vec(mean(S; dims = 2))
    nt = (y .== 0) .& (pat .== "NT")
    has(tok) = [any(occursin(tok, x) for x in c) for c in a.comment_tokens]
    groups = [("planets", y .== 1), ("FP: NT only", nt), ("FP: NT only, missed", nt .& (s .>= 0.5)),
              ("FP: NT only, caught", nt .& (s .< 0.5)), ("FP: MOD_NONUNIQ comment", (y .== 0) .& has("NONUNIQ")),
              ("FP: INDIV_TRANS / INCONSISTENT comment", (y .== 0) .& (has("INDIV_TRANS") .| has("INCONSISTENT"))),
              ("FP: EC only", (y .== 0) .& (pat .== "EC")), ("FP: SS only", (y .== 0) .& (pat .== "SS")),
              ("FP: CO only", (y .== 0) .& (pat .== "CO"))]
    q(v, m) = (x = filter(isfinite, Float64.(coalesce.(v[m], NaN))); isempty(x) ? NaN : round(median(x); sigdigits = 3))
    println("pool, medians per class:")
    show_df(DataFrame([(group = g, n = sum(m), (c => q(pool[!, c], m) for c in LC_NT_FEATURES)...) for (g, m) in groups]))
    rows = NamedTuple[]
    for c in LC_NT_FEATURES
        x = Float64.(coalesce.(pool[!, c], NaN))
        function au(m)
            m = m .& isfinite.(x); v = fast_auc(Float64.(y[m]), x[m]); return round(max(v, 1 - v); digits = 3)
        end
        push!(rows, (feature = string(c), finite_pct = round(100mean(isfinite.(x)); digits = 1), auc_dr25_alone = au(dr),
                     nt_only_vs_planets = au(dr .& ((y .== 1) .| nt)),
                     nt_missed_vs_planets = au((y .== 1) .| (nt .& (s .>= 0.5)))))
    end
    println("single-feature AUC (folded: 0.5 useless): DR25 pool; DR25 NT-only FPs vs planets; missed NT-only FPs vs planets (pool):")
    show_df(DataFrame(rows))
end
println(Dates.format(now(), "mm-dd HH:MM"), "  DONE")
