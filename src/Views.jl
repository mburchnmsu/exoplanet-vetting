# ==============================================================================
# VIEWS (Phase 2): fold a detrended light curve on a KOI's catalog ephemeris and
# bin it into fixed-length views (AstroNet style, Shallue & Vanderburg 2018).
#   global 2001 bins over the whole orbit; local / secondary / odd / even /
#   centroid 201 bins over ±2 transit durations (bin width 0.16 duration).
# Other KOIs on the same star are masked out first.
# ==============================================================================

const N_GLOBAL = 2001
const N_LOCAL  = 201
const LOCAL_SPAN = 2.0          # ± this many durations
const LOCAL_WIDTH = 0.16        # bin width, in durations

"""
    bin_median(phase, y, centres, width) -> (values, empty_fraction)

Median of `y` within ±width/2 of each centre (bins may overlap). Empty bins
are filled by linear interpolation between their neighbours.
"""
function bin_median(phase, y, centres, width)
    o = sortperm(phase); ph = phase[o]; yy = y[o]
    v = fill(NaN, length(centres))
    for (i, c) in enumerate(centres)
        a = searchsortedfirst(ph, c - width / 2); b = searchsortedlast(ph, c + width / 2)
        b >= a && (v[i] = median(@view yy[a:b]))
    end
    good = findall(isfinite, v)
    empty = 1 - length(good) / length(v)
    isempty(good) && return (zeros(length(v)), 1.0)
    for i in eachindex(v)
        isfinite(v[i]) && continue
        j = searchsortedfirst(good, i)
        v[i] = j > length(good) ? v[good[end]] : j == 1 ? v[good[1]] :
               v[good[j-1]] + (v[good[j]] - v[good[j-1]]) * (i - good[j-1]) / (good[j] - good[j-1])
    end
    return (v, empty)
end

local_centres(dur) = range(-LOCAL_SPAN * dur, LOCAL_SPAN * dur; length = N_LOCAL)
local_view(phase, y, dur) = bin_median(phase, y, local_centres(dur), LOCAL_WIDTH * dur)

"Phase of the deepest off-transit dip (box of one duration), or NaN if the orbit is too short to have one."
function secondary_phase(phase, f, P, dur)
    P < 4 * dur && return NaN
    o = sortperm(phase); ph = phase[o]; ff = f[o]
    ph2 = [ph .- P; ph; ph .+ P]; f2 = [ff; ff; ff]             # wrap around the orbit
    best, bestc = Inf, NaN
    for c in range(-P / 2, P / 2; step = dur / 4)
        abs(c) < 1.5 * dur && continue
        a = searchsortedfirst(ph2, c - dur / 2); b = searchsortedlast(ph2, c + dur / 2)
        b - a < 3 && continue
        m = mean(@view f2[a:b])
        m < best && ((best, bestc) = (m, c))
    end
    return bestc
end

"""
    koi_views(d, e, others) -> NamedTuple

All views for KOI ephemeris `e` from detrended light curve `d`, with the
transits of the star's `others` KOIs removed. Flux views are shifted so the
out-of-transit level is 0 and scaled so the local view's minimum is -1; odd,
even, secondary and global use the local view's scale, so relative depths
survive. Centroid views are in pixels relative to out of transit.
"""
function koi_views(d, e, others)
    keep = trues(length(d.time))
    for o in others
        keep .&= .!in_transit(d.time, o)
    end
    t, f = d.time[keep], d.flux[keep] .- 1
    ph = fold_phase.(t, e.period, e.t0)
    epoch = round.(Int, (t .- e.t0) ./ e.period)
    near = abs.(ph) .< e.dur / 2
    n_transits = length(unique(epoch[near]))
    oot = abs.(ph) .> e.dur
    base = any(oot) ? median(f[oot]) : 0.0
    f = f .- base

    loc, emp_local = local_view(ph, f, e.dur)
    scale = -minimum(loc)
    scale = scale > 0 ? scale : 1.0
    glo, emp_global = bin_median(ph, f, range(-e.period / 2, e.period / 2; length = N_GLOBAL), e.period / N_GLOBAL)
    odd = isodd.(epoch); ev = .!odd
    vodd, _ = any(odd) ? local_view(ph[odd], f[odd], e.dur) : (zeros(N_LOCAL), 1.0)
    veven, _ = any(ev) ? local_view(ph[ev], f[ev], e.dur) : (zeros(N_LOCAL), 1.0)
    sph = secondary_phase(ph, f, e.period, e.dur)
    vsec = isnan(sph) ? zeros(N_LOCAL) : first(local_view(fold_phase.(t, e.period, e.t0 + sph), f, e.dur))
    cen(c) = (ok = isfinite.(c); v = first(local_view(ph[ok], c[ok], e.dur));
              v .- median(v[abs.(local_centres(e.dur)) .> e.dur / 2 * 1.5]))
    return (name = e.name, global_view = glo ./ scale, local_view = loc ./ scale, odd = vodd ./ scale,
            even = veven ./ scale, secondary = vsec ./ scale, cen1 = cen(d.cen1[keep]), cen2 = cen(d.cen2[keep]),
            depth = scale, sec_phase = sph / e.period, n_transits, empty_local = emp_local,
            empty_global = emp_global, sigma = d.sigma)
end

"Detrend a star once and build views for each of its KOIs (names in `names`; all if empty)."
function star_views(kepid, df_raw; names = String[])
    ephs = star_ephemerides(df_raw, kepid)
    d = detrend_lc(load_lc(kepid), ephs)
    sel = isempty(names) ? ephs : filter(e -> e.name in names, ephs)
    return d, [koi_views(d, e, filter(o -> o.name != e.name, ephs)) for e in sel]
end
