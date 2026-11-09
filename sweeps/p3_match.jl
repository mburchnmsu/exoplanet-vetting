# Phase 3, M1: ephemeris-matching features for every KOI, plus the label-free null (random epochs).
#   (default)  all candidate matches, real and null (3 random-epoch draws); chance-match purity by separation and
#              ratio (no labels); the features -> sweeps/results/p3_ephemeris.csv
#   --classes  pool only: feature values per class (planets, EC-only, other FP patterns). Run after the feature
#              definitions are fixed from the label-free part.
#   --eb       also match against the Kepler EB catalog (v2) -> sweeps/results/p3_ephemeris_eb.csv
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p3_match.jl"
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))
using Dates

df_raw = load_koi()
const EB = "--eb" in ARGS
entries = ephemeris_entries(df_raw; eb = EB)
@printf("entries: %d (koi %d, tce %d, eb %d); targets: %d KOIs\n", nrow(entries), count(==("koi"), entries.catalog),
        count(==("tce"), entries.catalog), count(==("eb"), entries.catalog), nrow(df_raw))
out = joinpath(CV_DIR, EB ? "p3_ephemeris_eb.csv" : "p3_ephemeris.csv")
show_df(t) = (show(stdout, MIME"text/plain"(), t; allrows = true, allcols = true); println(); flush(stdout))
const SEPBINS = [(0, 10, "a < 10\""), (10, 60, "b 10-60\""), (60, 180, "c 1-3'"), (180, 600, "d 3-10'"),
                 (600, 1800, "e 10-30'"), (1800, Inf, "f > 30'")]
sepbin(x) = only(l for (a, b, l) in SEPBINS if a <= x < b)
rclass(m, n) = (m, n) == (1, 1) ? "1:1" : (m, n) in ((1, 2), (2, 1)) ? "1:2, 2:1" : "other"

if !("--classes" in ARGS)
    t0 = time()
    M = ephemeris_matches(df_raw, entries)
    @printf("real: %d candidate matches (score < 3) in %.1f s\n", nrow(M), time() - t0)
    nulls = [ephemeris_matches(df_raw, entries; rng = Random.Xoshiro(s)) for s in 1:3]
    # label-free purity: KOIs whose best score in a (separation, ratio) cell is < cut, real vs null
    function kois_below(D, cut)
        D = D[D.score .< cut, :]
        D.cell = string.(sepbin.(D.sep), " | ", rclass.(D.m, D.n))
        return Dict(key.cell => length(unique(g.kepoi_name)) for (key, g) in pairs(groupby(D, :cell)))
    end
    for cut in (1.0, 0.25)
        real = kois_below(M, cut)
        nl = [kois_below(d, cut) for d in nulls]
        rs = [(cell = c, real_kois = real[c], null_kois = round(mean(get(d, c, 0) for d in nl); digits = 1),
                 purity = round(1 - mean(get(d, c, 0) for d in nl) / real[c]; digits = 3)) for c in keys(real)]
        println("best score < $cut, by separation | ratio (purity = 1 − null/real):"); show_df(sort(DataFrame(rs), :cell))
    end
    F = ephemeris_features(df_raw, M)
    Fn = [ephemeris_features(df_raw, d) for d in nulls]
    rows = NamedTuple[]
    for (lab, f) in (("near score < 1", d -> d.eph_score_near .< 1), ("near score < 0.25", d -> d.eph_score_near .< 0.25),
                     ("far 1:1 score < 1", d -> d.eph_score_far .< 1), ("far 1:1 score < 0.25", d -> d.eph_score_far .< 0.25))
        r, nl = sum(f(F)), mean(sum(f(d)) for d in Fn)
        push!(rows, (feature = lab, real_kois = r, null_kois = round(nl; digits = 1), purity = round(1 - nl / r; digits = 3)))
    end
    println("features (simple ratios 1:1, 1:2, 2:1):"); show_df(DataFrame(rows))
    simple_near(D, cat) = length(unique(D.kepoi_name[(D.catalog .== cat) .& (D.sep .< NEAR) .& (D.score .< MATCH_CUT) .&
                                                     [(a, b) in SIMPLE for (a, b) in zip(D.m, D.n)]]))
    for cat in unique(entries.catalog)
        r, nl = simple_near(M, cat), mean(simple_near(d, cat) for d in nulls)
        @printf("near simple matches < %.2f against catalog %-4s: %d KOIs real, %.1f null (purity %.3f)\n",
                MATCH_CUT, cat, r, nl, 1 - nl / max(r, 1))
    end
    F.kepid = df_raw.kepid
    CSV.write(out, F)
    show_df(describe(F[:, EPHEMERIS_FEATURES], :mean, :median, :min, :max))
    println("saved: ", out)
end
if "--classes" in ARGS          # labels on the CV pool only; the test split is never looked at by class
    pool, _ = cv_pool(clean_koi(df_raw))
    F = CSV.read(out, DataFrame)
    pool = leftjoin(pool, F[:, [:kepoi_name; EPHEMERIS_FEATURES]]; on = :kepoi_name, order = :left)
    pat = with_vetting_info(pool, df_raw, zeros(nrow(pool))).flag_pattern
    ec_any = coalesce.(leftjoin(pool[:, [:kepoi_name]], df_raw[:, [:kepoi_name, :koi_fpflag_ec]];
                                on = :kepoi_name, order = :left).koi_fpflag_ec, 0) .== 1
    y = pool.label; dr = is_dr25(pool, df_raw)
    groups = [("planets", y .== 1), ("FP: EC only", (y .== 0) .& (pat .== "EC")), ("FP: EC flag (any)", (y .== 0) .& ec_any),
              ("FP: SS only", (y .== 0) .& (pat .== "SS")), ("FP: CO only", (y .== 0) .& (pat .== "CO")),
              ("FP: NT only", (y .== 0) .& (pat .== "NT")), ("FP: all", y .== 0)]
    rows = NamedTuple[]
    for (g, m) in groups
        near = coalesce.(pool.eph_score_near[m], 3.0) .< MATCH_CUT
        lk = filter(isfinite, coalesce.(pool.eph_logleak[m][near], NaN))
        push!(rows, (group = g, n = sum(m), near_match_pct = round(100mean(near); digits = 1),
                     far_match_pct = round(100mean(coalesce.(pool.eph_score_far[m], 3.0) .< MATCH_CUT); digits = 1),
                     leak_plausible_pct = round(100mean(lk .< 0); digits = 1)))
    end
    println("pool, per class: near (< $(Int(NEAR))\") and far 1:1 matches with score < $MATCH_CUT, and the share of near")
    println("matches where the other star could cause the dip (log leak < 0):"); show_df(DataFrame(rows))
    rows = NamedTuple[]
    for c in EPHEMERIS_FEATURES
        x = Float64.(coalesce.(pool[!, c], NaN)); m = dr .& isfinite.(x)
        a = fast_auc(Float64.(y[m]), x[m])
        push!(rows, (feature = string(c), auc_dr25_alone = round(max(a, 1 - a); digits = 3), planets_higher = a > 0.5,
                     ec_only_vs_planets = round((m2 = m .& ((y .== 1) .| (pat .== "EC")); a2 = fast_auc(Float64.(y[m2]), x[m2]);
                                                max(a2, 1 - a2)); digits = 3)))
    end
    println("single-feature AUC, DR25 pool (folded: 0.5 useless); last column = EC-only FPs vs planets:"); show_df(DataFrame(rows))
end
println(Dates.format(now(), "mm-dd HH:MM"), "  DONE")
