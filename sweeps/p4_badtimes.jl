# Phase 4, M0 (N-c): bad-time clustering features for every KOI (src/BadTimes.jl), plus the label-free null
# (all source TCE epochs shifted by a random fraction of their period: no clustering in time beyond chance).
#   (default)  global time series of other-star TCE midpoints, real vs null; the worst bins; KOI transits in bad
#              bins, real vs null; the features -> sweeps/results/p4_badtimes.csv
#   --classes  pool only: feature values per class and single-feature AUCs; agreement with the robovetter's SKYE
#              comment (analysis only, never an input). Run after the label-free part.
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p4_badtimes.jl [--classes]"
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
using Dates

df_raw = load_koi()
out = joinpath(CV_DIR, "p4_badtimes.csv")
show_df(t) = (show(stdout, MIME"text/plain"(), t; allrows = true, allcols = true); println(); flush(stdout))
r3(x) = round(x; digits = 3)

if !("--classes" in ARGS)
    t0 = time()
    src = badtime_sources()
    @printf("sources: %d DR25 TCEs with P >= %.0f d on %d stars; %d transit midpoints; %d bins of %.2f d\n",
            length(src.bins), BT_MINP, length(unique(src.kepid)), sum(length, src.bins), BT_NB, BT_BIN)
    F, info = badtime_features(df_raw, src)
    @printf("real: features in %.1f s; global median %.1f per 0.75-d window, σ %.2f\n", time() - t0, info.median, info.σ)
    nulls = [badtime_features(df_raw, badtime_sources(; rng = Random.Xoshiro(s))) for s in 1:3]
    rows = NamedTuple[]
    for (lab, F_, inf) in [("real", F, info); [("null $s", nulls[s]...) for s in 1:3]]
        push!(rows, (run = lab, bad_bins_z4 = sum(inf.z .> BT_ZCUT), bad_bins_z8 = sum(inf.z .> 8), max_z = r3(maximum(inf.z)),
                     kois_glob_frac_gt0 = sum(F_.bt_glob_frac .> 0), kois_glob_frac_ge05 = sum(F_.bt_glob_frac .>= 0.5),
                     kois_loc_frac_gt0 = sum(F_.bt_loc_frac .> 0), kois_loc_frac_ge05 = sum(F_.bt_loc_frac .>= 0.5)))
    end
    println("global bad bins and KOIs with transits in bad bins, real vs null (label-free):"); show_df(DataFrame(rows))
    o = sortperm(info.z; rev = true)[1:25]
    println("the 25 worst 0.75-d windows (BKJD centre, count of other-star TCE midpoints, z):")
    show_df(DataFrame(bkjd = r3.(BT_T0 .+ (o .- 0.5) .* BT_BIN), count = info.G[o], z = r3.(info.z[o])))
    for c in BADTIME_FEATURES
        @printf("%-13s real mean %.4f  null mean %.4f\n", c, mean(F[!, c]), mean(mean(n[1][!, c]) for n in nulls))
    end
    F.kepid = df_raw.kepid
    CSV.write(out, F)
    CSV.write(joinpath(CV_DIR, "p4_badtimes_series.csv"),
              DataFrame(bkjd = BT_T0 .+ ((1:BT_NB) .- 0.5) .* BT_BIN, count = info.G, z = info.z,
                        null_z = nulls[1][2].z))
    show_df(describe(F[:, BADTIME_FEATURES], :mean, :median, :min, :max))
    println("saved: ", out)
end
if "--classes" in ARGS          # labels on the CV pool only; the test split is never looked at by class
    pool, _ = cv_pool(clean_koi(df_raw))
    F = CSV.read(out, DataFrame)
    pool = leftjoin(pool, F[:, [:kepoi_name; BADTIME_FEATURES]]; on = :kepoi_name, order = :left)
    a = with_vetting_info(pool, df_raw, zeros(nrow(pool)))
    pat = a.flag_pattern
    y = pool.label; dr = is_dr25(pool, df_raw)
    skye = [any(occursin("SKYE", t) for t in c) for c in a.comment_tokens]
    S = (repeat_scores(load_cv("cv_mlpq_eph_r5s10.jld2")) .+ repeat_scores(load_cv("cv_evotree_eph_r5s10.jld2"))) ./ 2
    s = vec(mean(S; dims = 2))
    nt = (y .== 0) .& (pat .== "NT")
    groups = [("planets", y .== 1), ("FP: NT only", nt), ("FP: NT only, missed (p >= 0.5, + ephemeris)", nt .& (s .>= 0.5)),
              ("FP: NT only, caught", nt .& (s .< 0.5)), ("FP: SKYE comment (any)", (y .== 0) .& skye),
              ("FP: EC only", (y .== 0) .& (pat .== "EC")), ("FP: SS only", (y .== 0) .& (pat .== "SS")),
              ("FP: CO only", (y .== 0) .& (pat .== "CO")), ("FP: all", y .== 0)]
    rows = NamedTuple[]
    for (g, m) in groups
        push!(rows, (group = g, n = sum(m), glob_frac_gt0_pct = round(100mean(pool.bt_glob_frac[m] .> 0); digits = 1),
                     glob_frac_ge05_pct = round(100mean(pool.bt_glob_frac[m] .>= 0.5); digits = 1),
                     loc_frac_gt0_pct = round(100mean(pool.bt_loc_frac[m] .> 0); digits = 1),
                     loc_frac_ge05_pct = round(100mean(pool.bt_loc_frac[m] .>= 0.5); digits = 1),
                     glob_mean_med = r3(median(pool.bt_glob_mean[m])), loc_mean_med = r3(median(pool.bt_loc_mean[m]))))
    end
    println("pool, per class: share with any / at least half of their transits in bad bins:"); show_df(DataFrame(rows))
    rows = NamedTuple[]
    for c in BADTIME_FEATURES
        x = Float64.(pool[!, c])
        au(m) = (q = fast_auc(Float64.(y[m]), x[m]); r3(max(q, 1 - q)))
        push!(rows, (feature = string(c), auc_dr25_alone = au(dr), planets_higher = fast_auc(Float64.(y[dr]), x[dr]) > 0.5,
                     nt_only_vs_planets = au(dr .& ((y .== 1) .| nt)), skye_vs_rest_fps = (m = y .== 0;
                     q = fast_auc(Float64.(skye[m]), x[m]); r3(q))))
    end
    println("single-feature AUC, DR25 pool (folded: 0.5 useless); NT-only FPs vs planets; SKYE-comment FPs vs other FPs (unfolded):")
    show_df(DataFrame(rows))
end
println(Dates.format(now(), "mm-dd HH:MM"), "  DONE")
