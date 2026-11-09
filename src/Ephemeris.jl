# ==============================================================================
# EPHEMERIS MATCHING (Phase 3): does a KOI's signal match a signal on ANOTHER star?
# A contaminating source (direct PSF overlap, column bleed, cross-talk) imprints
# its own period and transit times on the victim. We search the KOI table and the
# DR25 TCE list (and the Kepler EB catalog, when available) for such matches.
# Label-free by construction (PHASE3_PLAN.md, E4/E5): only ephemeris, depth,
# duration, position and brightness of the other objects are used, never their
# dispositions or flags, and a target is never matched to its own star.
# ==============================================================================

const CATALOG_DIR = joinpath(DATA_DIR, "catalogs")
const MISSION_SPAN = 1460.0        # days, Q1–Q17 (BKJD ~131.5 to ~1591)
const ALL_RATIOS = sort(unique([(m ÷ gcd(m, n), n ÷ gcd(m, n)) for m in 1:10, n in 1:10]))

"""
    ephemeris_entries(df_raw; eb = false) -> DataFrame

Every signal that could be a contamination source: all KOIs (catalog = "koi") and all DR25 TCEs ("tce"), with
kepid, period [d], epoch [BKJD], duration [h], depth [ppm], ra, dec [deg], kepmag. Only these columns are read.
With `eb = true`, also the Kepler EB catalog ("eb"; period, epoch and the star's position and magnitude only; its
depth and duration are unknown, NaN). v1 (M1/M2) ran without it.
"""
function ephemeris_entries(df_raw::DataFrame; eb = false)
    k = DataFrame(catalog = "koi", kepid = df_raw.kepid, period = df_raw.koi_period, epoch = df_raw.koi_time0bk,
                  duration = df_raw.koi_duration, depth = df_raw.koi_depth, ra = df_raw.ra, dec = df_raw.dec,
                  kepmag = df_raw.koi_kepmag)
    t = CSV.read(joinpath(CATALOG_DIR, "dr25_tce.csv"), DataFrame)
    st = CSV.read(joinpath(CATALOG_DIR, "dr25_stellar.csv"), DataFrame)
    mag = Dict(zip(st.kepid, st.kepmag))
    t = DataFrame(catalog = "tce", kepid = t.kepid, period = t.tce_period, epoch = t.tce_time0bk,
                  duration = t.tce_duration, depth = t.tce_depth, ra = t.ra, dec = t.dec,
                  kepmag = [get(mag, id, missing) for id in t.kepid])
    e = vcat(k, t)
    if eb
        # VizieR TSV: comment lines, a header, a units row and a dashes row, then KIC  Per  BJD0  Kpmag
        num(s) = (v = tryparse(Float64, strip(s)); v === nothing ? NaN : v)
        rs = [split(l, '\t') for l in eachline(joinpath(CATALOG_DIR, "kepler_eb.tsv")) if !isempty(l) && isdigit(l[1])]
        b = DataFrame(KIC = [parse(Int, r[1]) for r in rs], Per = [num(r[2]) for r in rs], BJD0 = [num(r[3]) for r in rs])
        pos = Dict(zip(st.kepid, zip(st.ra, st.dec)))
        b = filter(r -> haskey(pos, r.KIC), b)
        e = vcat(e, DataFrame(catalog = "eb", kepid = b.KIC, period = b.Per,
                              epoch = coalesce.(b.BJD0, NaN) .- 54833.0,          # BJD − 2400000 -> BKJD
                              duration = NaN, depth = NaN, ra = [pos[i][1] for i in b.KIC],
                              dec = [pos[i][2] for i in b.KIC], kepmag = [get(mag, i, missing) for i in b.KIC]))
    end
    for c in [:period, :epoch, :duration, :depth, :ra, :dec, :kepmag]
        e[!, c] = Float64.(coalesce.(e[!, c], NaN))
    end
    filter!(r -> isfinite(r.period) && r.period > 0 && isfinite(r.epoch), e)
    return sort!(e, :period)
end

"Angular separation [arcsec] between (ra1, dec1) and (ra2, dec2) [deg] (haversine)."
function sep_arcsec(ra1, dec1, ra2, dec2)
    φ1, φ2, Δφ, Δλ = deg2rad(dec1), deg2rad(dec2), deg2rad(dec2 - dec1), deg2rad(ra2 - ra1)
    a = sin(Δφ / 2)^2 + cos(φ1) * cos(φ2) * sin(Δλ / 2)^2
    return rad2deg(2 * asin(min(1.0, sqrt(a)))) * 3600
end

"""
    match_score(PA, TA, dA, PB, TB, m, n) -> (score, drift, offset)

For the hypothesis P_A = (m/n)·P_B, in units of A's duration dA [days]:
- drift: how far the two ephemerides walk apart over the mission, (span / P_A)·|P_A − (m/n)·P_B|;
- offset: the smallest gap between A's transits and B's, which is (T_A − T_B) wrapped to ±P_A/(2m)
  (with gcd(m, n) = 1, coincidences recur every P_A/m).
score = √(drift² + offset²); below ~1 the two signals coincide for the whole mission.
"""
function match_score(PA, TA, dA, PB, TB, m, n)
    drift = MISSION_SPAN / PA * abs(PA - m / n * PB) / dA
    w = PA / m
    off = abs(mod(TA - TB + w / 2, w) - w / 2) / dA
    return (score = hypot(drift, off), drift = drift, offset = off)
end

"""
    ephemeris_matches(targets, entries; ratios = ALL_RATIOS, maxscore = 3.0, rng = nothing) -> DataFrame

All candidate matches (score < `maxscore`) between each target KOI (needs kepid, kepoi_name, koi_period,
koi_time0bk, koi_duration, koi_depth, ra, dec, koi_kepmag) and the entries on OTHER stars, one row per pair and
ratio: kepoi_name, catalog, score, sep [arcsec], dmag (other − target), logleak (log10 of the leak fraction the
other signal would need to cause the target's depth; < 0 = plausible source), ratio m:n, other (its kepid).
With `rng`, every entry's epoch is first shifted by a random fraction of its period: the null (chance only).
"""
function ephemeris_matches(targets::DataFrame, entries::DataFrame; ratios = ALL_RATIOS, maxscore = 3.0, rng = nothing)
    e = copy(entries)
    rng === nothing || (e.epoch = e.epoch .+ rand(rng, nrow(e)) .* e.period)
    P = e.period
    M = DataFrame(kepoi_name = String[], catalog = String[], score = Float64[], sep = Float64[], dmag = Float64[],
                  logleak = Float64[], m = Int[], n = Int[], other = Int[])
    for r in eachrow(targets)
        PA, TA, dA = Float64(r.koi_period), Float64(r.koi_time0bk), Float64(r.koi_duration) / 24
        (isfinite(PA) && isfinite(TA) && isfinite(dA) && dA > 0) || continue
        tol = maxscore * dA * PA / MISSION_SPAN               # |P_A − (m/n)·P_B| allowed by the drift cut
        magA = Float64(coalesce(r.koi_kepmag, NaN)); depA = max(Float64(coalesce(r.koi_depth, NaN)), 1.0)
        for (m, n) in ratios
            for j in searchsortedfirst(P, (PA - tol) * n / m):searchsortedlast(P, (PA + tol) * n / m)
                e.kepid[j] == r.kepid && continue             # never the target's own star (E5)
                s = match_score(PA, TA, dA, P[j], e.epoch[j], m, n).score
                s < maxscore || continue
                dmag = e.kepmag[j] - magA
                # leak fraction f with depth_B·f·F_B/F_A = depth_A  ->  f = (δA/δB)·10^(0.4·(magB − magA))
                push!(M, (r.kepoi_name, e.catalog[j], s, sep_arcsec(r.ra, r.dec, e.ra[j], e.dec[j]), dmag,
                          clamp(log10(depA / max(e.depth[j], 1.0)) + 0.4 * dmag, -6.0, 6.0), m, n, e.kepid[j]))
            end
        end
    end
    return M
end

const NO_MATCH = 3.0               # score when nothing matches (= the search cut; not a missing indicator)
const NEAR = 600.0                 # arcsec: "near" matches (label-free null: chance matches are rare inside)
const SIMPLE = [(1, 1), (1, 2), (2, 1)]
const MATCH_CUT = 0.25             # a "match": label-free null shows ~74% of near matches below it are real (11/04)
const EPHEMERIS_FEATURES = [
    :eph_score_near,  # best score among near (< NEAR) matches at ratios 1:1, 1:2, 2:1 [durations; NO_MATCH if none]
    :eph_logsep,      # log10 separation of that match [arcsec] (log10 NEAR if none)
    :eph_logleak,     # its log10 leak fraction (< 0: the other star could cause our dip; 6 if none)
    :eph_dmag,        # its Kepler magnitude − ours (0 if none)
    :eph_nmatch_near, # other near stars matched with score < MATCH_CUT
    :eph_score_far,   # best 1:1 score beyond NEAR (cross-talk / column effects; mostly chance, the model decides)
]

"Collapse matches to one feature row per target (targets without any match get the 'none' values)."
function ephemeris_features(targets::DataFrame, M::DataFrame)
    F = DataFrame(kepoi_name = targets.kepoi_name, eph_score_near = NO_MATCH, eph_logsep = log10(NEAR),
                  eph_logleak = 6.0, eph_dmag = 0.0, eph_nmatch_near = 0.0, eph_score_far = NO_MATCH)
    idx = Dict(n => i for (i, n) in enumerate(F.kepoi_name))
    simple = [(a, b) in SIMPLE for (a, b) in zip(M.m, M.n)]
    for g in groupby(M[simple, :], :kepoi_name)
        i = idx[g.kepoi_name[1]]
        near = g[g.sep .< NEAR, :]
        if nrow(near) > 0
            k = argmin(near.score)
            F.eph_score_near[i] = near.score[k]; F.eph_logsep[i] = log10(max(near.sep[k], 0.1))
            F.eph_logleak[i] = near.logleak[k]; F.eph_dmag[i] = isfinite(near.dmag[k]) ? near.dmag[k] : 0.0
            if near.catalog[k] == "eb"            # EB depths are unknown: take the leak from the best match that has one
                kk = findall(isfinite, near.logleak)
                isempty(kk) || (F.eph_logleak[i] = near.logleak[kk[argmin(near.score[kk])]])
            end
            F.eph_nmatch_near[i] = length(unique(near.other[near.score .< MATCH_CUT]))
        end
        far = g[(g.sep .>= NEAR) .& (g.m .== 1) .& (g.n .== 1), :]
        nrow(far) > 0 && (F.eph_score_far[i] = minimum(far.score))
    end
    return F
end
