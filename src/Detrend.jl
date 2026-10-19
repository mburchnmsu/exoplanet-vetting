# ==============================================================================
# DETRENDING (Phase 2): quality masking, per-quarter normalisation, in-transit
# masks for every KOI on the star, and a robust sliding Tukey-biweight trend
# (the default filter of `wotan`, Hippke et al. 2019), written out in Julia.
# ==============================================================================

# lightkurve's "default" Kepler bitmask: attitude tweak, safe mode, coarse point,
# Earth point, desaturation, manual exclude (1+2+4+8+32+128).
const QUALITY_MASK = Int32(175)

"Ephemerides of every KOI on a star: (period [d], t0 [BKJD], duration [d]). From the raw catalog."
function star_ephemerides(df_raw::DataFrame, kepid)
    r = df_raw[(df_raw.kepid .== kepid) .& .!ismissing.(df_raw.koi_period) .& .!ismissing.(df_raw.koi_time0bk) .&
               .!ismissing.(df_raw.koi_duration), :]
    return [(name = n, period = Float64(p), t0 = Float64(t), dur = Float64(d) / 24)
            for (n, p, t, d) in zip(r.kepoi_name, r.koi_period, r.koi_time0bk, r.koi_duration)]
end

"Phase in days, in [-P/2, P/2), with the transit at 0."
fold_phase(t, P, t0) = mod(t - t0 + P / 2, P) - P / 2

"Bool mask: within `factor` × duration / 2 of a transit centre of `e`."
in_transit(t, e; factor = 1.5) = abs.(fold_phase.(t, e.period, e.t0)) .< factor * e.dur / 2

"Tukey biweight location of `x` (c = 5, as in wotan), a few iterations from the median."
function biweight_location(x; c = 5.0, iters = 5)
    M = median(x)
    for _ in 1:iters
        mad = median(abs.(x .- M))
        mad == 0 && return M
        num = den = 0.0
        for xi in x
            u = (xi - M) / (c * mad)
            abs(u) < 1 || continue
            w = (1 - u^2)^2
            num += w * xi; den += w
        end
        Mn = num / den
        abs(Mn - M) < 1e-7 * abs(M) && return Mn
        M = Mn
    end
    return M
end

"""
    biweight_trend(t, f, use; window) -> trend

Sliding biweight over a time window (days), evaluated at every point but fitted
only to points with `use == true` (in-transit points excluded). NaN where the
window holds fewer than 5 usable points.
"""
function biweight_trend(t, f, use; window)
    n = length(t); trend = fill(NaN, n)
    lo = hi = 1
    for i in 1:n
        while t[lo] < t[i] - window / 2; lo += 1; end
        while hi < n && t[hi + 1] <= t[i] + window / 2; hi += 1; end
        idx = [j for j in lo:hi if use[j]]
        length(idx) >= 5 && (trend[i] = biweight_location(@view f[idx]))
    end
    return trend
end

"Segments split at quarter boundaries and at gaps longer than `gap` days (wotan's break tolerance)."
function segments(t, quarter; gap = 0.5)
    cuts = [0; findall((diff(t) .> gap) .| (diff(quarter) .!= 0)); length(t)]
    return [cuts[i]+1:cuts[i+1] for i in 1:length(cuts)-1]
end

"""
    detrend_lc(lc, ephs; factor = 1.5, clip = 3.0) -> NamedTuple

Clean and flatten a cached light curve. Drops flagged / non-finite cadences,
normalises each quarter by its median, masks every KOI's transits, divides by
a biweight trend with window max(0.5 d, 3 × the longest KOI duration), and
clips upward outliers only (> `clip` robust σ). Returns time, flux (≈ 1 out of
transit), the raw normalised flux and trend (for plots), centroids with their
own trend removed, quarter, and the window used.
"""
function detrend_lc(lc, ephs; factor = 1.5, clip = 3.0)
    ok = ((lc.quality .& QUALITY_MASK) .== 0) .& isfinite.(lc.flux) .& isfinite.(lc.time)
    t, q = lc.time[ok], lc.quarter[ok]
    raw = Float64.(lc.flux[ok]); c1 = Float64.(lc.cen1[ok]); c2 = Float64.(lc.cen2[ok])
    for qq in unique(q)
        s = q .== qq
        raw[s] ./= median(raw[s])
    end
    intr = isempty(ephs) ? falses(length(t)) : reduce(.|, (in_transit(t, e; factor) for e in ephs))
    window = max(0.5, 3 * maximum(e.dur for e in ephs; init = 0.0))
    trend = fill(NaN, length(t)); ct1 = fill(NaN, length(t)); ct2 = fill(NaN, length(t))
    for s in segments(t, q)
        use = .!intr[s]
        trend[s] = biweight_trend(t[s], raw[s], use; window)
        u1 = use .& isfinite.(c1[s]); u2 = use .& isfinite.(c2[s])
        ct1[s] = biweight_trend(t[s], c1[s], u1; window); ct2[s] = biweight_trend(t[s], c2[s], u2; window)
    end
    flux = raw ./ trend
    oot = .!intr .& isfinite.(flux)
    σ = 1.4826 * median(abs.(flux[oot] .- median(flux[oot])))
    keep = isfinite.(flux) .& (flux .< 1 + clip * σ)                      # never clip downward
    return (time = t[keep], flux = flux[keep], raw = raw[keep], trend = trend[keep], quarter = q[keep],
            cen1 = (c1 .- ct1)[keep], cen2 = (c2 .- ct2)[keep], sigma = σ, window)
end
