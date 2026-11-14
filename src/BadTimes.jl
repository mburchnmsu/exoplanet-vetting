# ==============================================================================
# BAD-TIME CLUSTERING (Phase 4, N-c; like the robovetter's SKYE test): do a KOI's
# transits fall on days when an excess of OTHER stars' DR25 TCE transits fall too?
# Rolling-band noise, thermal settling after downlinks, momentum dumps and safe
# modes make spurious events at the same times on many stars, so the predicted
# transit midpoints of such TCEs pile up in time. Label-free: only the TCE list's
# periods, epochs and positions are used (never dispositions, flags or scores),
# and a target's own star never counts towards its excess.
# ==============================================================================

const BT_T0, BT_T1 = 120.0, 1600.0       # BKJD range covering Q1–Q17 (~131.5 to ~1591)
const BT_BIN = 0.25                     # days per time bin
const BT_NB = ceil(Int, (BT_T1 - BT_T0) / BT_BIN)
const BT_MINP = 1.0                     # sources: TCEs with P >= 1 d (shorter ones fill every bin, no time information)
const BT_LOCAL = 1800.0                 # arcsec: "local" sources (~ the same CCD channel in any quarter)
const BT_ZCUT, BT_SCUT = 4.0, 3.0       # bad bin: global robust z > 4; local Poisson −log10 p > 3
const BADTIME_FEATURES = [
    :bt_glob_mean,  # mean over the KOI's predicted transits of the global robust z of other-star TCE midpoints (clamped −5…20)
    :bt_glob_frac,  # fraction of its transits in globally bad bins (z > BT_ZCUT)
    :bt_loc_mean,   # mean over its transits of the local (< BT_LOCAL) Poisson significance −log10 p (clamped 0…10)
    :bt_loc_frac,   # fraction of its transits in locally bad bins (−log10 p > BT_SCUT)
]

"Time-bin index of each predicted transit midpoint of (P, T0) inside the mission range (unique, sorted)."
function transit_bins(P, T0)
    (isfinite(P) && isfinite(T0) && P > 0) || return Int32[]
    k0, k1 = ceil(Int, (BT_T0 - T0) / P), floor(Int, (BT_T1 - T0) / P)
    return unique!(Int32[clamp(floor(Int, (T0 + k * P - BT_T0) / BT_BIN) + 1, 1, BT_NB) for k in k0:k1])
end

"3-bin running sum (a 0.75-d window centred on each bin), so events near a bin edge are not split."
smooth3(c) = [c[b] + (b > 1 ? c[b - 1] : 0) + (b < length(c) ? c[b + 1] : 0) for b in eachindex(c)]

"Poisson upper-tail significance −log10 P(X ≥ k | λ)."
function poisson_sig(k, λ)
    k <= λ && return 0.0
    # P(X >= k) = 1 − Σ_{i<k} e^{-λ} λ^i / i!, summed in log space for stability
    lp = -λ + k * log(λ) - sum(log, 1:k; init = 0.0)    # log P(X = k)
    term, s = 0.0, 1.0                                  # Σ_{j>=0} P(k+j)/P(k)
    for j in 1:200
        term += log(λ / (k + j)); s += exp(term)
        exp(term) < 1e-12 && break
    end
    return -(lp + log(s)) / log(10)
end

"""
    badtime_sources(; rng = nothing) -> NamedTuple

The DR25 TCEs used as event sources (P >= BT_MINP): kepid, ra, dec and their transit bins. With `rng`, every
epoch is shifted by a random fraction of its period first (the null: no time clustering beyond chance).
"""
function badtime_sources(; rng = nothing)
    t = CSV.read(joinpath(CATALOG_DIR, "dr25_tce.csv"), DataFrame)
    t = t[(coalesce.(t.tce_period, 0.0) .>= BT_MINP) .& .!ismissing.(t.tce_time0bk), :]
    T0 = Float64.(t.tce_time0bk)
    rng === nothing || (T0 = T0 .+ rand(rng, nrow(t)) .* t.tce_period)
    return (kepid = Int.(t.kepid), ra = Float64.(t.ra), dec = Float64.(t.dec),
            bins = [transit_bins(Float64(p), t0) for (p, t0) in zip(t.tce_period, T0)])
end

"""
Per-bin counts of source midpoints (3-bin smoothed) for the sources `idx`. `weighted`: each TCE has total weight 1
spread over its transits (1/N each), so short-period TCEs, which fill every bin, do not swamp the time structure
(label-free check 11/13: unweighted, the global series shows no excess over the null; weighted, 77 windows with
z > 4 vs ~1 in the null).
"""
function bin_counts(src, idx; weighted = false)
    c = zeros(BT_NB)
    for j in idx, b in src.bins[j]
        c[b] += weighted ? 1 / length(src.bins[j]) : 1.0
    end
    return smooth3(c)
end

"""
    badtime_features(targets, src) -> (F, info)

BADTIME_FEATURES for every row of `targets` (kepoi_name, kepid, koi_period, koi_time0bk, ra, dec). Global: all
sources, 1/N-weighted, robust z per bin against the median and MAD over bins, with the target star's own TCEs
removed. Local: sources on other stars within BT_LOCAL, Poisson significance against that neighbourhood's
mean count per bin. `info` has the global series (counts, z) for the label-free checks.
"""
function badtime_features(targets::DataFrame, src)
    G = bin_counts(src, eachindex(src.bins); weighted = true)
    m = median(G); σ = 1.4826 * median(abs.(G .- m))
    byid = Dict{Int, Vector{Int}}()
    for (j, id) in enumerate(src.kepid)
        push!(get!(byid, id, Int[]), j)
    end
    F = DataFrame(kepoi_name = targets.kepoi_name, bt_glob_mean = 0.0, bt_glob_frac = 0.0, bt_loc_mean = 0.0,
                  bt_loc_frac = 0.0)
    cosd_ = cosd.(src.dec)
    for g in groupby(DataFrame(i = 1:nrow(targets), kepid = targets.kepid), :kepid)
        r = targets[g.i[1], :]
        own = bin_counts(src, get(byid, r.kepid, Int[]); weighted = true)
        zg = (G .- own .- m) ./ σ
        ra0, dec0 = Float64(r.ra), Float64(r.dec)
        near = [j for j in eachindex(src.kepid) if src.kepid[j] != r.kepid && abs(src.dec[j] - dec0) < BT_LOCAL / 3600 &&
                sep_arcsec(ra0, dec0, src.ra[j], src.dec[j]) < BT_LOCAL]
        L = bin_counts(src, near)
        λ = max(mean(L), 1e-3)
        sl = Dict{Int, Float64}()
        for i in g.i
            tb = transit_bins(Float64(coalesce(targets.koi_period[i], NaN)), Float64(coalesce(targets.koi_time0bk[i], NaN)))
            isempty(tb) && continue
            s = [get!(() -> poisson_sig(L[b], λ), sl, b) for b in tb]
            F.bt_glob_mean[i] = mean(clamp.(zg[tb], -5.0, 20.0)); F.bt_glob_frac[i] = mean(zg[tb] .> BT_ZCUT)
            F.bt_loc_mean[i] = mean(clamp.(s, 0.0, 10.0));      F.bt_loc_frac[i] = mean(s .> BT_SCUT)
        end
    end
    return F, (G = G, z = (G .- m) ./ σ, median = m, σ = σ)
end
