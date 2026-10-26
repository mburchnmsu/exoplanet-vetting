# Phase 2, M3 part 1: light-curve features (src/LCFeatures.jl) for every KOI on every cached star.
# Resumable: shards of 250 stars in data/lightcurves/features/shard_NNN.jld2; finished shards are skipped.
# Output: sweeps/results/p2_lcfeatures.csv (one row per KOI, no labels; lc_timing and lc_ntransits joined from
# M2's p2_qa.csv, which used the same detrending).
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p2_features.jl"
#   ... sweeps/p2_features.jl --test 40     time the first 40 stars, print a few rows, save nothing
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
using Dates

const FEAT_DIR = joinpath(LC_DIR, "features")
const SHARD = 250

df_raw = load_koi()
stars = sort(unique(clean_koi(df_raw).kepid))
shard_path(i) = joinpath(FEAT_DIR, @sprintf("shard_%03d.jld2", i))

function feat_star(k)
    try
        return star_lc_features(k, df_raw), ""
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

mkpath(FEAT_DIR)
nsh = cld(length(stars), SHARD)
t_start = time(); done_now = 0
for i in 1:nsh
    isfile(shard_path(i)) && continue
    ks = stars[(i-1)*SHARD+1:min(i * SHARD, end)]
    out = run_stars(ks)
    feats = reduce(vcat, (first(o) for o in out); init = Any[])
    errs = [(kepid = k, error = last(o)) for (k, o) in zip(ks, out) if !isempty(last(o))]
    jldsave(shard_path(i) * ".part"; feats, errs)
    mv(shard_path(i) * ".part", shard_path(i); force = true)
    global done_now += 1
    el = (time() - t_start) / 3600
    left = count(j -> !isfile(shard_path(j)), 1:nsh)
    @printf("%s  shard %d/%d: %d KOIs, %d star errors; %.2f h elapsed, ~%.1f h left\n", Dates.format(now(), "HH:MM"),
            i, nsh, length(feats), length(errs), el, el / done_now * left)
    flush(stdout)
end

feats = Any[]; errs = Any[]
for i in 1:nsh
    d = JLD2.load(shard_path(i))
    append!(feats, d["feats"]); append!(errs, d["errs"])
end
lc = DataFrame(feats)
qa = CSV.read(joinpath(CV_DIR, "p2_qa.csv"), DataFrame)
lc = leftjoin(lc, select(qa, :kepoi_name, :timing_scatter => :lc_timing, :n_transits => :lc_ntransits);
              on = :kepoi_name, order = :left)
CSV.write(joinpath(CV_DIR, "p2_lcfeatures.csv"), lc)
isempty(errs) || CSV.write(joinpath(CV_DIR, "p2_features_errors.csv"), DataFrame(errs))
@printf("\nM3 features: %d KOIs on %d stars, %d star errors\n", nrow(lc), length(stars), length(errs))
for c in [LC_FEATURES; LC_CENTROID_FEATURES]
    v = Float64.(coalesce.(lc[!, c], NaN)); ok = filter(isfinite, v)
    @printf("  %-15s finite %5d / %d   median %9.3g   [5%%, 95%%] %9.3g %9.3g\n", c, length(ok), length(v),
            median(ok), quantile(ok, 0.05), quantile(ok, 0.95))
end
