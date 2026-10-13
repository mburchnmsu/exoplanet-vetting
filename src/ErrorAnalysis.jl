# ==============================================================================
# ERROR ANALYSIS: where and why the vetter is wrong, using out-of-fold scores.
# Vetting columns (fpflags, koi_score, koi_comment) are attached here for
# ANALYSIS ONLY. They never enter feature_matrix (check_no_leakage enforces it).
# Included automatically by VettingSetup.jl.
# ==============================================================================

const FLAG_COLS  = [:koi_fpflag_nt, :koi_fpflag_ss, :koi_fpflag_co, :koi_fpflag_ec]
const FLAG_NAMES = Dict(:koi_fpflag_nt => "NT (not transit-like)", :koi_fpflag_ss => "SS (stellar eclipse)",
                        :koi_fpflag_co => "CO (centroid offset)", :koi_fpflag_ec => "EC (ephemeris match)")
const ANALYSIS_COLS = [FLAG_COLS; :koi_score; :koi_comment; :koi_pdisposition; :koi_tce_delivname]

"Join the analysis-only vetting columns onto `df` by kepoi_name, plus the score `s`."
function with_vetting_info(df::DataFrame, df_raw::DataFrame, s)
    out = leftjoin(df, df_raw[:, [:kepoi_name; ANALYSIS_COLS]]; on = :kepoi_name, order = :left)
    @assert out.kepoi_name == df.kepoi_name
    out.score = s
    for c in FLAG_COLS
        out[!, c] = coalesce.(out[!, c], 0)
    end
    out.flag_pattern = [flag_pattern(r) for r in eachrow(out)]
    out.comment_tokens = [ismissing(c) ? String[] : filter(!isempty, split(c, "---")) .|> String for c in out.koi_comment]
    return out
end

function flag_pattern(r)
    on = [split(FLAG_NAMES[c])[1] for c in FLAG_COLS if r[c] == 1]
    return isempty(on) ? "none" : join(on, "+")
end

"""
    fp_flag_table(a; threshold = 0.5)

For false positives: per robovetter flag (overlapping) and per exact flag
combination, how many there are, their median score, and the fraction the
model wrongly calls planets (score >= threshold).
"""
function fp_flag_table(a::DataFrame; threshold = 0.5)
    fp = a[a.label .== 0, :]
    rows = NamedTuple[]
    push!(rows, (group = "all FPs", n = nrow(fp), median_score = median(fp.score),
                 missed_frac = mean(fp.score .>= threshold)))
    for c in FLAG_COLS
        m = fp[!, c] .== 1
        push!(rows, (group = "has " * FLAG_NAMES[c], n = sum(m), median_score = median(fp.score[m]),
                     missed_frac = mean(fp.score[m] .>= threshold)))
    end
    by = combine(groupby(fp, :flag_pattern), nrow => :n, :score => median => :median_score,
                 :score => (s -> mean(s .>= threshold)) => :missed_frac)
    sort!(by, :n; rev = true)
    for r in eachrow(by)
        push!(rows, (group = "only " * r.flag_pattern, n = r.n, median_score = r.median_score, missed_frac = r.missed_frac))
    end
    return DataFrame(rows)
end

"""
    comment_token_table(a; threshold = 0.5, min_n = 40)

Robovetter comment codes (e.g. CENT_RESOLVED_OFFSET, DEEP_V_SHAPED) among false
positives: how often the model misses FPs carrying each code. Finer-grained
than the four flags.
"""
function comment_token_table(a::DataFrame; threshold = 0.5, min_n = 40)
    fp = a[a.label .== 0, :]
    toks = unique(reduce(vcat, fp.comment_tokens))
    rows = NamedTuple[]
    for t in toks
        m = [t in ts for ts in fp.comment_tokens]
        sum(m) >= min_n || continue
        push!(rows, (token = t, n = sum(m), median_score = median(fp.score[m]),
                     missed_frac = mean(fp.score[m] .>= threshold)))
    end
    return sort!(DataFrame(rows), :missed_frac; rev = true)
end

"""
    binned_errors(a, feature; nbins = 10, threshold = 0.5, log = false)

Quantile bins of a raw feature. Per bin: KOI count, planet fraction,
FP-miss rate (FPs called planets), planet-miss rate (planets called FPs), and
ROC-AUC within the bin (how well the model ranks KOIs that look alike in this
parameter).
"""
function binned_errors(a::DataFrame, feature::Symbol; nbins = 10, threshold = 0.5)
    x = Float64.(a[!, feature])
    edges = unique(quantile(x, range(0, 1; length = nbins + 1)))
    b = clamp.(searchsortedlast.(Ref(edges), x), 1, length(edges) - 1)
    rows = NamedTuple[]
    for i in 1:length(edges)-1
        m = b .== i
        y, s = a.label[m], a.score[m]
        np, nn = sum(y .== 1), sum(y .== 0)
        push!(rows, (feature = feature, bin = i, lo = edges[i], hi = edges[i+1], center = median(x[m]),
                     n = sum(m), planet_frac = mean(y),
                     fp_miss = nn > 0 ? mean(s[y .== 0] .>= threshold) : NaN,
                     planet_miss = np > 0 ? mean(s[y .== 1] .< threshold) : NaN,
                     auc = (np >= 10 && nn >= 10) ? fast_auc(y, s) : NaN))
    end
    return DataFrame(rows)
end

"Plot binned FP-miss / planet-miss rates and within-bin AUC for several features."
function plot_binned_errors(a::DataFrame, features; nbins = 10, threshold = 0.5, logx = Symbol[], ncols = 3)
    nrows_ = cld(length(features), ncols)
    fig = Figure(size = (330ncols, 270nrows_ + 90))
    for (i, f) in enumerate(features)
        t = binned_errors(a, f; nbins, threshold)
        ax = Axis(fig[cld(i, ncols), mod1(i, ncols)]; title = string(f), ylabel = i % ncols == 1 ? "rate" : "",
                  xscale = f in logx ? log10 : identity, limits = (nothing, (0, 1)))
        scatterlines!(ax, t.center, t.fp_miss; label = "FP called planet", color = :firebrick)
        scatterlines!(ax, t.center, t.planet_miss; label = "planet called FP", color = :steelblue)
        scatterlines!(ax, t.center, t.planet_frac; label = "planet fraction", color = :gray, linestyle = :dash)
        scatterlines!(ax, t.center, t.auc; label = "AUC in bin", color = :darkgreen, marker = :diamond)
        i == 1 && Legend(fig[nrows_ + 1, :], ax; orientation = :horizontal, nbanks = 2, framevisible = false)
    end
    Label(fig[0, :], "Errors by parameter (quantile bins, threshold $(threshold))"; fontsize = 16, font = :bold)
    return fig
end

"Score distributions by disposition (CONFIRMED / CANDIDATE / FALSE POSITIVE)."
function plot_score_by_disposition(a::DataFrame; title = "Out-of-fold planet score by disposition")
    fig = Figure(size = (750, 420))
    ax = Axis(fig[1, 1]; xlabel = "planet score (OOF, ensemble)", ylabel = "density", title)
    for (d, col) in (("FALSE POSITIVE", :firebrick), ("CANDIDATE", :darkorange), ("CONFIRMED", :steelblue))
        s = a.score[a.koi_disposition .== d]
        hist!(ax, s; bins = 0:0.025:1, normalization = :pdf, color = (col, 0.35), strokewidth = 1,
              strokecolor = col, label = "$d (n=$(length(s)))")
    end
    axislegend(ax; position = :ct)
    return fig
end

"Score distributions for FPs split by flag (overlapping groups)."
function plot_score_by_flag(a::DataFrame)
    fp = a[a.label .== 0, :]
    fig = Figure(size = (750, 420))
    ax = Axis(fig[1, 1]; xlabel = "planet score (OOF, ensemble)", ylabel = "cumulative fraction of FPs",
              title = "False positives: how planet-like each flag type looks")
    for (c, col) in zip(FLAG_COLS, (:purple, :firebrick, :darkorange, :teal))
        s = sort(fp.score[fp[!, c] .== 1])
        stairs!(ax, s, (1:length(s)) ./ length(s); label = FLAG_NAMES[c] * " (n=$(length(s)))", color = col, linewidth = 2)
    end
    vlines!(ax, [0.5]; color = :gray, linestyle = :dash)
    axislegend(ax; position = :rb)
    return fig
end

"""
    koi_score_comparison(a; threshold = 0.5)

Analysis-only comparison with the robovetter's koi_score (missing for ~16% of
KOIs). koi_score comes from the same vetting that set the dispositions, so it is
partly circular with the labels: a sanity check, not a benchmark.
"""
function koi_score_comparison(a::DataFrame; threshold = 0.5)
    b = a[.!ismissing.(a.koi_score), :]
    ks = Float64.(b.koi_score)
    agree = (ks .>= threshold) .== (b.score .>= threshold)
    summary = (n = nrow(b), n_missing = nrow(a) - nrow(b),
               spearman = corspearman(ks, b.score),
               agree_frac = mean(agree),
               auc_model = fast_auc(b.label, b.score), auc_koi_score = fast_auc(b.label, ks))
    return summary, b
end

function plot_koi_score_comparison(b::DataFrame)
    fig = Figure(size = (900, 420))
    for (j, (d, ttl)) in enumerate((("FALSE POSITIVE", "False positives"), ("planet", "CONFIRMED + CANDIDATE")))
        m = d == "planet" ? b.label .== 1 : b.label .== 0
        ax = Axis(fig[1, j]; xlabel = "robovetter koi_score", ylabel = j == 1 ? "our OOF score" : "", title = ttl, aspect = 1)
        h = fit(Histogram, (Float64.(b.koi_score[m]), b.score[m]), (0:0.05:1.0001, 0:0.05:1.0001))
        heatmap!(ax, 0.025:0.05:1, 0.025:0.05:1, log10.(h.weights .+ 1); colormap = :viridis)
        lines!(ax, [0, 1], [0, 1]; color = :white, linestyle = :dash)
        limits!(ax, 0, 1, 0, 1)
    end
    Colorbar(fig[1, 3]; colormap = :viridis, label = "log10(count + 1)", limits = (0, 3))
    return fig
end
