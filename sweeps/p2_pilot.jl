# Phase 2, M0 pilot: ~50 POOL stars covering each class (no test stars). Downloads their light curves into the
# compact cache, detrends them, builds every view for every KOI on those stars, and times each stage so the full
# M1 download can be extrapolated. Labels/flags are used only to CHOOSE the pilot stars and to name them.
#   julia --project=. -t 2 sweeps/p2_pilot.jl
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

df_raw = load_koi()
df = clean_koi(df_raw)
pool, test = cv_pool(df)
info = leftjoin(pool[:, [:kepid, :kepoi_name, :label, :koi_disposition, :koi_period]],
                df_raw[:, [:kepoi_name, :koi_fpflag_nt, :koi_fpflag_ss, :koi_fpflag_co, :koi_fpflag_ec]]; on = :kepoi_name)
info.dr25 = is_dr25(info, df_raw)
fp(r) = r.label == 0
cats = [("CONFIRMED",       r -> r.dr25 && r.koi_disposition == "CONFIRMED", 8),
        ("CANDIDATE",       r -> r.dr25 && r.koi_disposition == "CANDIDATE", 6),
        ("FP: SS (EB)",     r -> r.dr25 && fp(r) && r.koi_fpflag_ss == 1 && r.koi_fpflag_co == 0 && r.koi_fpflag_nt == 0, 7),
        ("FP: CO (offset)", r -> r.dr25 && fp(r) && r.koi_fpflag_co == 1 && r.koi_fpflag_ss == 0, 7),
        ("FP: NT",          r -> r.dr25 && fp(r) && r.koi_fpflag_nt == 1 && r.koi_fpflag_ss == 0 && r.koi_fpflag_co == 0, 7),
        ("FP: EC",          r -> r.dr25 && fp(r) && r.koi_fpflag_ec == 1, 4),
        ("USP planet",      r -> r.dr25 && r.label == 1 && r.koi_period < 1, 3),
        ("Long-P planet",   r -> r.dr25 && r.label == 1 && r.koi_period > 200, 3),
        ("Supplemental planet", r -> !r.dr25 && r.label == 1, 2),
        ("Supplemental FP",     r -> !r.dr25 && r.label == 0, 2)]
rng = Xoshiro(2026)
used = Set{Int}()
picks = DataFrame(kepid = Int[], kepoi_name = String[], category = String[])
for (name, pred, n) in cats
    cand = shuffle(rng, findall(r -> coalesce(pred(r), false) && !(r.kepid in used), eachrow(info)))
    for i in cand[1:min(n, end)]
        push!(picks, (info.kepid[i], info.kepoi_name[i], name)); push!(used, info.kepid[i])
    end
end
@assert isempty(intersect(Set(picks.kepid), Set(test.kepid)))
CSV.write(joinpath(CV_DIR, "p2_pilot_stars.csv"), picks)
println(nrow(picks), " pilot stars")

# 1. download -> compact cache
t0 = time()
download_stars(picks.kepid; ntasks = 4, log_every = 10)
t_dl = time() - t0
man = filter(r -> r.kepid in used, CSV.read(MANIFEST, DataFrame))
ok = man[man.status .== "ok", :]

# 2. detrend + views for every KOI on each pilot star
t0 = time()
views = Dict{String, Any}()
tdet = Float64[]
for k in unique(ok.kepid)
    t1 = time()
    _, vs = star_views(k, df_raw)
    push!(tdet, time() - t1)
    for v in vs
        views[v.name] = v
    end
end
t_views = time() - t0
jldsave(joinpath(LC_DIR, "pilot_views.jld2"); views)

# 3. extrapolate to all stars of the cleaned catalog (pool + test)
nstars = length(unique(df.kepid))
f = nrow(ok) / max(nrow(man), 1)
per_star_dl = sum(ok.seconds) / max(nrow(ok), 1)
println("\n=== M0 pilot summary ===")
@printf("stars ok %d / %d (errors %d), quarters per star median %.0f\n", nrow(ok), nrow(man),
        nrow(man) - nrow(ok), median(ok.nfiles))
@printf("raw FITS per star %.1f MB, cache per star %.2f MB (%.0f%% of raw)\n", mean(ok.raw_bytes) / 1e6,
        mean(ok.cache_bytes) / 1e6, 100 * sum(ok.cache_bytes) / sum(ok.raw_bytes))
@printf("download+cache %.1f s per star; detrend+views %.1f s per star (%d KOIs)\n", per_star_dl, mean(tdet), length(views))
@printf("extrapolated to %d stars: raw %.0f GB, cache %.1f GB, download %.1f h, views %.1f h (1 process)\n", nstars,
        nstars * mean(ok.raw_bytes) / 1e9, nstars * mean(ok.cache_bytes) / 1e9, nstars * per_star_dl / 3600,
        nstars * mean(tdet) / 3600)
CSV.write(joinpath(CV_DIR, "p2_pilot_timing.csv"),
          DataFrame(stars_ok = nrow(ok), stars_tried = nrow(man), raw_mb_per_star = mean(ok.raw_bytes) / 1e6,
                    cache_mb_per_star = mean(ok.cache_bytes) / 1e6, dl_s_per_star = per_star_dl,
                    views_s_per_star = mean(tdet), n_kois = length(views), all_stars = nstars,
                    est_raw_gb = nstars * mean(ok.raw_bytes) / 1e9, est_cache_gb = nstars * mean(ok.cache_bytes) / 1e9,
                    est_dl_h = nstars * per_star_dl / 3600, est_views_h = nstars * mean(tdet) / 3600))
