# Phase 2, M2: detrend every cached star and build all views + QA columns for every KOI on it.
# Resumable: stars are processed in shards of 250 (data/lightcurves/views/shard_NNN.jld2); finished shards are
# skipped. At the end the shards are assembled into data/views_all.jld2 (Float32 view matrices, git-ignored) and
# sweeps/results/p2_qa.csv (one row per KOI; no labels in it).
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p2_build.jl"
#   ... sweeps/p2_build.jl --test 40     time the first 40 stars, save nothing
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
using Dates

const SHARD_DIR = joinpath(LC_DIR, "views")
const SHARD = 250
const VIEW_KEYS = (:global_view, :local_view, :odd, :even, :secondary, :cen1, :cen2)
const QA_KEYS = (:depth, :sec_phase, :n_transits, :empty_local, :empty_global, :sigma, :timing_scatter, :max_single_ratio)

df_raw = load_koi()
stars = sort(unique(clean_koi(df_raw).kepid))
shard_path(i) = joinpath(SHARD_DIR, @sprintf("shard_%03d.jld2", i))

"Views for one star, arrays as Float32; (results, error message)."
function build_star(k)
    try
        _, vs = star_views(k, df_raw)
        return [(; kepid = k, (key => (key in VIEW_KEYS ? Float32.(getfield(v, key)) : getfield(v, key))
                                for key in keys(v))...) for v in vs], ""
    catch e
        return [], first(replace(sprint(showerror, e), r"[\r\n]" => " "), 200)
    end
end

function run_stars(ks)
    out = Vector{Any}(undef, length(ks))
    Threads.@threads :dynamic for i in eachindex(ks)
        out[i] = build_star(ks[i])
    end
    return out
end

if "--test" in ARGS
    n = parse(Int, ARGS[findfirst(==("--test"), ARGS) + 1])
    run_stars(stars[1:2])                                     # compile
    GC.gc(); m0 = Base.gc_live_bytes()
    t0 = time(); out = run_stars(stars[3:n+2]); dt = time() - t0
    @printf("test: %d stars, %d KOIs, %d errors, %.2f s/star wall with %d threads -> all %d stars ~%.1f h; RSS %.2f GB\n",
            n, sum(length(first(o)) for o in out), count(o -> !isempty(last(o)), out), dt / n, Threads.nthreads(),
            length(stars), dt / n * length(stars) / 3600, Sys.maxrss() / 1e9)
    exit()
end

mkpath(SHARD_DIR)
nsh = cld(length(stars), SHARD)
t_start = time(); done_now = 0
for i in 1:nsh
    isfile(shard_path(i)) && continue
    ks = stars[(i-1)*SHARD+1:min(i * SHARD, end)]
    out = run_stars(ks)
    views = reduce(vcat, (first(o) for o in out); init = Any[])
    errs = [(kepid = k, error = last(o)) for (k, o) in zip(ks, out) if !isempty(last(o))]
    jldsave(shard_path(i) * ".part"; views, errs)
    mv(shard_path(i) * ".part", shard_path(i); force = true)
    global done_now += 1
    el = (time() - t_start) / 3600
    left = count(j -> !isfile(shard_path(j)), 1:nsh)
    @printf("%s  shard %d/%d: %d KOIs, %d star errors; %.2f h elapsed, ~%.1f h left\n", Dates.format(now(), "HH:MM"),
            i, nsh, length(views), length(errs), el, el / done_now * left)
    flush(stdout)
end

# --- assemble ---------------------------------------------------------------------------
views = Any[]; errs = Any[]
for i in 1:nsh
    d = JLD2.load(shard_path(i))
    append!(views, d["views"]); append!(errs, d["errs"])
end
names_ = [String(v.name) for v in views]
mats = Dict(k => reduce(hcat, [getfield(v, k) for v in views]) for k in VIEW_KEYS)
jldsave(joinpath(DATA_DIR, "views_all.jld2"); names = names_, kepid = [v.kepid for v in views], mats...)

qa = DataFrame([:kepid => [v.kepid for v in views]; :kepoi_name => names_;
                [k => [getfield(v, k) for v in views] for k in QA_KEYS]])
cat = df_raw[:, [:kepoi_name, :koi_depth, :koi_duration]]
qa = leftjoin(qa, cat; on = :kepoi_name, order = :left)
qa.depth_ppm = 1e6 .* qa.depth
qa.depth_ratio = coalesce.(qa.depth_ppm ./ qa.koi_depth, NaN)
# per-transit SNR from the CATALOG depth (our own depth is noise-dominated for the weakest signals); 0.02043 d = 1 cadence
qa.snr1 = coalesce.(1e-6 .* qa.koi_depth ./ qa.sigma .* sqrt.(qa.koi_duration ./ 24 ./ 0.02043), NaN)
select!(qa, Not(:koi_duration))
qa.qa_few_transits = qa.n_transits .< 2
qa.qa_empty = qa.empty_local .> 0.5
qa.qa_depth_mismatch = coalesce.((qa.depth_ratio .< 0.5) .| (qa.depth_ratio .> 2), false)
qa.qa_timing = (qa.snr1 .>= 7) .& (qa.timing_scatter .> 0.25)    # only where single transits are clearly detected
CSV.write(joinpath(CV_DIR, "p2_qa.csv"), qa)
isempty(errs) || CSV.write(joinpath(CV_DIR, "p2_build_errors.csv"), DataFrame(errs))
@printf("\nM2 build: %d KOIs with views on %d stars, %d star errors. QA flags: few transits %d, empty %d, depth mismatch %d, timing %d\n",
        nrow(qa), length(unique(qa.kepid)), length(errs), sum(qa.qa_few_transits), sum(qa.qa_empty),
        sum(qa.qa_depth_mismatch), sum(qa.qa_timing))
