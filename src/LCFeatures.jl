# ==============================================================================
# LIGHT-CURVE FEATURES (Phase 2a): a few hand-made diagnostics per KOI, computed
# from our own detrended light curve (independent of the robovetter's numbers).
# Every signal is expressed as a SIGNIFICANCE against an empirical noise level:
# the scatter of one-duration box means away from the transit, which includes
# red noise on the transit timescale (M2 showed raw view values are noise-dominated).
# ==============================================================================

const LC_FEATURES = [
    :lc_snr,          # our transit depth / box noise (signal that survives our detrending)
    :lc_depth_ratio,  # trapezoid-fit depth / catalog depth
    :lc_vshape,       # trapezoid ingress fraction: 0 = box (planet-like), 1 = V (EB-like)
    :lc_dur_ratio,    # trapezoid total duration / catalog duration
    :lc_oe_sig,       # odd vs even depth difference [σ]
    :lc_sec_sig,      # deepest off-transit dip (secondary eclipse) [σ]
    :lc_sec_ratio,    # its depth / the primary depth
    :lc_sec_phase,    # its phase, |Δφ| in [0, 0.5]
    :lc_sev_ratio,    # strongest single transit's SNR / total SNR (≈ 1/√N for N equal transits, ≈ 1 for one glitch)
    :lc_timing,       # robust scatter of per-transit centre offsets [durations] (from M2's QA pass)
    :lc_ntransits,    # observed transits
]
const LC_CENTROID_FEATURES = [
    :lc_cen_sig,      # in-transit centroid shift [σ], both axes combined
    :lc_cen_offset,   # shift / fractional depth [pixels]: ~ distance of the eclipsing source from the target
]

"Phase-sorted copy of a folded series, wrapped by ±P, with prefix sums: O(log n) box means."
function box_index(ph, y, P)
    o = sortperm(ph); p = ph[o]; v = y[o]
    return (p = [p .- P; p; p .+ P], cs = [0.0; cumsum([v; v; v])])
end

"(mean, count) of the points within ±w/2 of phase c."
function box_mean(B, c, w)
    a = searchsortedfirst(B.p, c - w / 2); b = searchsortedlast(B.p, c + w / 2)
    n = b - a + 1
    return n > 0 ? ((B.cs[b+1] - B.cs[a]) / n, n) : (NaN, 0)
end

"""
    box_noise(B, P, dur) -> (centres, means, counts, level, σ_eff)

One-duration boxes every dur/4, at least min(3 dur, P/4) from the transit. `level` is their median (the baseline),
`σ_eff` the per-point noise that reproduces their scatter (MAD of mean × √count), so a box of n points has
noise σ_eff / √n. Fewer than 10 boxes -> NaN level and σ_eff.
"""
function box_noise(B, P, dur; min_pts = 3)
    excl = min(3 * dur, P / 4)
    cs = Float64[]; ms = Float64[]; ns = Int[]
    P < 2 * dur && return cs, ms, ns, NaN, NaN
    for c in range(-P / 2, P / 2; step = dur / 4)
        abs(c) < excl && continue
        m, n = box_mean(B, c, dur)
        n >= min_pts || continue
        push!(cs, c); push!(ms, m); push!(ns, n)
    end
    length(ms) < 10 && return cs, ms, ns, NaN, NaN
    lvl = median(ms)
    return cs, ms, ns, lvl, 1.4826 * median(abs.((ms .- lvl) .* sqrt.(ns)))
end

"Trapezoid (centred, depth a > 0) fitted by grid search over total duration T and ingress fraction r to binned points."
function trapezoid_fit(x, y, w, dur)
    best = (Inf, NaN, NaN, NaN)
    s = similar(x)
    for T in (0.2:0.05:3.0) .* dur, r in 0:0.05:1
        hf = T / 2; fl = hf * (1 - r)
        @. s = ifelse(abs(x) <= fl, 1.0, ifelse(abs(x) < hf, (hf - abs(x)) / (hf - fl), 0.0))
        sw = sum(w .* s .^ 2)
        sw > 0 || continue
        a = -sum(w .* y .* s) / sw
        a > 0 || continue
        chi = sum(w .* (y .+ a .* s) .^ 2)
        chi < best[1] && (best = (chi, a, T / dur, r))
    end
    return (depth = best[2], dur_ratio = best[3], vshape = best[4])
end

"""
    koi_lc_features(d, e, others, koi_depth_ppm) -> NamedTuple

Light-curve features of KOI ephemeris `e` from detrended light curve `d` (see `detrend_lc`), with the other KOIs on
the star masked out. `lc_timing` and `lc_ntransits` come from M2's QA table and are joined later.
"""
function koi_lc_features(d, e, others, koi_depth_ppm)
    keep = trues(length(d.time))
    for o in others
        keep .&= .!in_transit(d.time, o)
    end
    t, f = d.time[keep], d.flux[keep] .- 1
    P, dur = e.period, e.dur
    ph = fold_phase.(t, P, e.t0)
    B = box_index(ph, f, P)
    cs, ms, ns, lvl, σe = box_noise(B, P, dur)
    if !isfinite(σe)                     # orbit too short for a box scan: white noise from out-of-transit points
        oot = abs.(ph) .> dur
        lvl = any(oot) ? median(f[oot]) : 0.0
        σe = any(oot) ? 1.4826 * median(abs.(f[oot] .- lvl)) : NaN
    end
    m0, n0 = box_mean(B, 0.0, dur)
    dep0 = lvl - m0
    snr = dep0 / (σe / sqrt(max(n0, 1)))

    # trapezoid on mean-binned points within ±2 durations (bins of 0.05 duration, weighted by count)
    near2 = abs.(ph) .< 2 * dur
    edges = range(-2 * dur, 2 * dur; length = 81)
    xb = collect((edges[1:end-1] .+ edges[2:end]) ./ 2); yb = zeros(80); wb = zeros(80)
    for (p, v) in zip(ph[near2], f[near2])
        i = clamp(searchsortedlast(edges, p), 1, 80)
        yb[i] += v - lvl; wb[i] += 1
    end
    yb = ifelse.(wb .> 0, yb ./ max.(wb, 1), 0.0)
    tz = trapezoid_fit(xb, yb, wb, dur)
    depth = isfinite(tz.depth) ? tz.depth : dep0

    # odd / even
    epoch = round.(Int, (t .- e.t0) ./ P)
    intr = abs.(ph) .< dur / 2
    io, ie = intr .& isodd.(epoch), intr .& iseven.(epoch)
    oe = any(io) && any(ie) ?
         abs(mean(f[io]) - mean(f[ie])) / (σe * sqrt(1 / count(io) + 1 / count(ie))) : NaN

    # secondary: the deepest off-transit box
    sec_sig = sec_ratio = sec_phase = NaN
    if length(ms) >= 10
        i = argmin(ms)
        sdep = lvl - ms[i]
        sec_sig = sdep / (σe / sqrt(ns[i]))
        sec_ratio = depth > 0 ? sdep / depth : NaN
        sec_phase = abs(cs[i]) / P
    end

    # single-event dominance: strongest transit's SNR vs the total
    sums = Dict{Int, Tuple{Float64, Int}}()
    for (n, v) in zip(epoch[intr], f[intr])
        s, k = get(sums, n, (0.0, 0)); sums[n] = (s + v, k + 1)
    end
    sev = length(sums) >= 2 && dep0 > 0 ?
          maximum((lvl - s / k) * sqrt(k) for (s, k) in values(sums)) / (dep0 * sqrt(n0)) : NaN

    # centroids (+ centroid family only): in-transit shift per axis, against the same kind of box noise
    function cen_shift(c)
        ok = isfinite.(c)
        count(ok) > 20 || return (NaN, NaN)
        Bc = box_index(ph[ok], c[ok], P)
        _, _, _, lc, σc = box_noise(Bc, P, dur)
        mc, nc = box_mean(Bc, 0.0, dur)
        isfinite(σc) && σc > 0 && nc > 0 || return (NaN, NaN)
        return (mc - lc, (mc - lc) / (σc / sqrt(nc)))
    end
    s1, z1 = cen_shift(d.cen1[keep]); s2, z2 = cen_shift(d.cen2[keep])

    return (kepoi_name = e.name, lc_snr = snr,
            lc_depth_ratio = koi_depth_ppm > 0 ? 1e6 * depth / koi_depth_ppm : NaN,
            lc_vshape = tz.vshape, lc_dur_ratio = tz.dur_ratio, lc_oe_sig = oe,
            lc_sec_sig = sec_sig, lc_sec_ratio = sec_ratio, lc_sec_phase = sec_phase, lc_sev_ratio = sev,
            lc_cen_sig = hypot(z1, z2), lc_cen_offset = depth > 0 ? hypot(s1, s2) / depth : NaN)
end

# ------------------------------------------------------------------------------
# Phase 4 (N-a, N-b): is the event UNIQUE in its own light curve, and do the
# individual transits AGREE? Aimed at "not transit-like" FPs (PHASE4_PLAN.md).
# Same detrending and box-noise level σ_eff as the Phase 2 features.
# ------------------------------------------------------------------------------
const LC_NT_FEATURES = [
    :nt_red,       # red-noise factor: σ_eff (box scatter, transit timescale) / point-to-point noise (1 = white) [N-a]
    :nt_pos_sig,   # strongest positive bump at any off-transit phase [σ]                                    [N-a]
    :nt_ter_sig,   # strongest off-transit dip away from the secondary (the "tertiary") [σ]                  [N-a]
    :nt_ms_ter,    # lc_snr − nt_ter_sig: primary vs the next-strongest unrelated dip (model-shift MS1-like)  [N-a]
    :nt_ms_pos,    # lc_snr − nt_pos_sig: primary vs the strongest bump (MS3-like)                           [N-a]
    :nt_chi2,      # log10 reduced χ² of the individual transit depths about their weighted mean            [N-b]
    :nt_negfrac,   # fraction of individual transits with depth < 0 (SNR below 0)                           [N-b]
    :nt_chases,    # median over transits of SNR_i / max(SNR_i, strongest local dip within ±5 durations)     [N-b]
    :nt_rubble,    # median over transits of observed / expected in-transit cadences                         [N-b]
]
const CADENCE = 0.020434                # days, Kepler long cadence

"""
    koi_nt_features(d, e, others) -> NamedTuple

LC_NT_FEATURES of KOI ephemeris `e` from detrended light curve `d`, the other KOIs on the star masked out.
"""
function koi_nt_features(d, e, others)
    keep = trues(length(d.time))
    for o in others
        keep .&= .!in_transit(d.time, o)
    end
    t, f = d.time[keep], d.flux[keep] .- 1
    P, dur = e.period, e.dur
    nanrow = (; kepoi_name = e.name, (c => NaN for c in LC_NT_FEATURES)...)
    length(t) > 100 && dur > 0 || return nanrow
    ph = fold_phase.(t, P, e.t0)
    B = box_index(ph, f, P)
    cs, ms, ns, lvl, σe = box_noise(B, P, dur)
    oot = abs.(ph) .> dur
    σw = count(oot) > 20 ? 1.4826 * median(abs.(diff(f[oot]))) / sqrt(2) : NaN     # point-to-point white noise
    if !isfinite(σe)
        lvl = any(oot) ? median(f[oot]) : 0.0
        σe = any(oot) ? 1.4826 * median(abs.(f[oot] .- lvl)) : NaN
    end
    isfinite(σe) && σe > 0 || return nanrow
    m0, n0 = box_mean(B, 0.0, dur)
    snr = (lvl - m0) / (σe / sqrt(max(n0, 1)))

    # N-a: uniqueness in the folded light curve (off-transit boxes from box_noise)
    pos = ter = NaN
    if length(ms) >= 10
        sig = (lvl .- ms) ./ (σe ./ sqrt.(ns))                  # dip significance of every off-transit box
        isec = argmax(sig)
        pos = maximum(-sig)
        away = [abs(mod(c - cs[isec] + P / 2, P) - P / 2) > 2 * dur for c in cs]
        ter = any(away) ? maximum(sig[away]) : NaN
    end

    # N-b: individual transits
    epoch = round.(Int, (t .- e.t0) ./ P)
    intr = abs.(ph) .< dur / 2
    sums = Dict{Int, Tuple{Float64, Int}}()
    for (n, v) in zip(epoch[intr], f[intr])
        s, k = get(sums, n, (0.0, 0)); sums[n] = (s + v, k + 1)
    end
    eps = sort!(collect(keys(sums)))
    dep = [lvl - sums[n][1] / sums[n][2] for n in eps]
    err = [σe / sqrt(sums[n][2]) for n in eps]
    chi2 = negfrac = NaN
    if length(eps) >= 2
        w = 1 ./ err .^ 2
        dm = sum(w .* dep) / sum(w)
        chi2 = log10(max(sum(w .* (dep .- dm) .^ 2) / (length(eps) - 1), 1e-3))
        negfrac = mean(dep .< 0)
    end
    exp_n = dur / CADENCE
    rubble = isempty(eps) ? NaN : median(min(sums[n][2] / exp_n, 1.5) for n in eps)

    # chases: each transit against the strongest one-duration dip in its own neighbourhood (time domain)
    cs_t = [0.0; cumsum(f)]
    function tbox(c)
        a = searchsortedfirst(t, c - dur / 2); b = searchsortedlast(t, c + dur / 2)
        n = b - a + 1
        return n >= 3 ? ((lvl - (cs_t[b+1] - cs_t[a]) / n) / (σe / sqrt(n)), n) : (NaN, 0)
    end
    ch = Float64[]
    for (n, de, er) in zip(eps, dep, err)
        si = de / er
        si > 0 || continue
        tc = e.t0 + n * P
        loc = -Inf
        for c in range(tc - 5dur, tc + 5dur; step = dur / 4)
            abs(c - tc) < 1.5dur && continue
            s, k = tbox(c)
            k > 0 && isfinite(s) && (loc = max(loc, s))
        end
        isfinite(loc) && push!(ch, si / max(si, loc))
    end
    chases = isempty(ch) ? NaN : median(ch)

    return (kepoi_name = e.name, nt_red = isfinite(σw) && σw > 0 ? σe / σw : NaN, nt_pos_sig = pos, nt_ter_sig = ter,
            nt_ms_ter = snr - ter, nt_ms_pos = snr - pos, nt_chi2 = chi2, nt_negfrac = negfrac, nt_chases = chases,
            nt_rubble = rubble)
end

"Phase 4: detrend a star once (flux only; the same detrending as Phase 2) and compute LC_NT_FEATURES of every KOI on it."
function star_nt_features(kepid, df_raw)
    ephs = star_ephemerides(df_raw, kepid)
    lc = load_lc(kepid)
    lc = merge(lc, (cen1 = fill(NaN32, length(lc.time)), cen2 = fill(NaN32, length(lc.time))))   # skip centroid trends
    d = detrend_lc(lc, ephs)
    return [koi_nt_features(d, e, filter(o -> o.name != e.name, ephs)) for e in ephs]
end

# ------------------------------------------------------------------------------
# Phase 4 (N-d): does the signal survive other detrendings? A real transit keeps
# its depth when the trend is fitted differently; a dip made by variability or by
# the trend interpolated across the masked transit window changes or vanishes.
# ------------------------------------------------------------------------------
const LC_ND_FEATURES = [
    :nd_dep_short,   # depth with an UNMASKED short biweight (window max(0.25 d, 2.5 durations)) / default depth
    :nd_snr_short,   # SNR with that detrending / default SNR
    :nd_dep_long,    # depth with a masked long biweight (window max(2 d, 10 × the longest duration)) / default depth
    :nd_snr_long,    # SNR with that detrending / default SNR
]

"(depth, SNR) of KOI `e` in detrended light curve `d`, other KOIs masked; the same box statistics as koi_lc_features."
function box_depth_snr(d, e, others)
    keep = trues(length(d.time))
    for o in others
        keep .&= .!in_transit(d.time, o)
    end
    t, f = d.time[keep], d.flux[keep] .- 1
    length(t) > 100 && e.dur > 0 || return (NaN, NaN)
    ph = fold_phase.(t, e.period, e.t0)
    B = box_index(ph, f, e.period)
    _, _, _, lvl, σe = box_noise(B, e.period, e.dur)
    if !isfinite(σe)
        oot = abs.(ph) .> e.dur
        lvl = any(oot) ? median(f[oot]) : 0.0
        σe = any(oot) ? 1.4826 * median(abs.(f[oot] .- lvl)) : NaN
    end
    m0, n0 = box_mean(B, 0.0, e.dur)
    dep = lvl - m0
    return dep, dep / (σe / sqrt(max(n0, 1)))
end

"Phase 4: LC_ND_FEATURES of every KOI on a star (default, unmasked-short per KOI, and masked-long detrendings)."
function star_nd_features(kepid, df_raw)
    ephs = star_ephemerides(df_raw, kepid)
    lc = load_lc(kepid)
    lc = merge(lc, (cen1 = fill(NaN32, length(lc.time)), cen2 = fill(NaN32, length(lc.time))))   # skip centroid trends
    d0 = detrend_lc(lc, ephs)
    dl = detrend_lc(lc, ephs; window = max(2.0, 10 * maximum(e.dur for e in ephs; init = 0.0)))
    ratio(a, b) = isfinite(a) && isfinite(b) && abs(b) > 0 ? clamp(a / b, -3.0, 3.0) : NaN
    rows = NamedTuple[]
    for e in ephs
        others = filter(o -> o.name != e.name, ephs)
        ds = detrend_lc(lc, others; window = max(0.25, 2.5 * e.dur))       # the target itself is NOT masked
        p0, ps, pl = box_depth_snr(d0, e, others), box_depth_snr(ds, e, others), box_depth_snr(dl, e, others)
        push!(rows, (kepoi_name = e.name, nd_dep_short = ratio(ps[1], p0[1]), nd_snr_short = ratio(ps[2], p0[2]),
                     nd_dep_long = ratio(pl[1], p0[1]), nd_snr_long = ratio(pl[2], p0[2])))
    end
    return rows
end

"Detrend a star once and compute the LC features of every KOI on it."
function star_lc_features(kepid, df_raw)
    ephs = star_ephemerides(df_raw, kepid)
    d = detrend_lc(load_lc(kepid), ephs)
    dep = Dict(r.kepoi_name => coalesce(r.koi_depth, NaN) for r in eachrow(df_raw[df_raw.kepid .== kepid, :]))
    return [koi_lc_features(d, e, filter(o -> o.name != e.name, ephs), dep[e.name]) for e in ephs]
end
