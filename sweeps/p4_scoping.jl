# Phase 4 scoping (pool only, CV out-of-fold scores, no new models): which "not transit-like" (NT) false positives
# does the best catalog-family model (Phase 3 "+ ephemeris", MLP + trees 50/50) still miss? Robovetter comments are
# read here for ANALYSIS ONLY (grouping), never as inputs. Feeds PHASE4_PLAN.md.
#   julia --project=. -t 2 sweeps/p4_scoping.jl
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

df_raw = load_koi()
pool, _ = cv_pool(clean_koi(df_raw))
pool = leftjoin(pool, CSV.read(joinpath(CV_DIR, "p2_lcfeatures.csv"), DataFrame); on = :kepoi_name, order = :left)
y = Float64.(pool.label)
dr = is_dr25(pool, df_raw)
a = with_vetting_info(pool, df_raw, zeros(nrow(pool)))
S = (repeat_scores(load_cv("cv_mlpq_eph_r5s10.jld2")) .+ repeat_scores(load_cv("cv_evotree_eph_r5s10.jld2"))) ./ 2
s = vec(mean(S; dims = 2))                                     # mean OOF score over fold shuffles
show_df(t) = (show(stdout, MIME"text/plain"(), t; allrows = true, allcols = true); println(); flush(stdout))
nt = (y .== 0) .& (a.flag_pattern .== "NT")
pl = y .== 1
missed = nt .& (s .>= 0.5)
@printf("NT-only FPs: %d (DR25 %d); missed at p = 0.5: %d (%.1f%%; DR25 %.1f%%, supplemental %.1f%%)\n", sum(nt), sum(nt .& dr),
        sum(missed), 100mean(s[nt] .>= 0.5), 100mean(s[nt .& dr] .>= 0.5), 100mean(s[nt .& .!dr] .>= 0.5))

# 1. robovetter comment tokens among NT-only FPs: caught vs missed (analysis only)
toks = unique(reduce(vcat, a.comment_tokens[nt]))
rows = NamedTuple[]
for t in toks
    has = [t in c for c in a.comment_tokens]
    n = sum(nt .& has); n >= 15 || continue
    push!(rows, (token = t, n_nt = n, missed_pct = round(100mean(s[nt .& has] .>= 0.5); digits = 1),
                 share_of_missed = round(100sum(missed .& has) / sum(missed); digits = 1)))
end
println("robovetter comment tokens on NT-only FPs (n >= 15):"); show_df(sort(DataFrame(rows), :n_nt; rev = true))

# 2. what the missed NT FPs look like vs planets and caught NT FPs (medians)
q(v, m) = round(median(filter(isfinite, Float64.(coalesce.(v[m], NaN)))); sigdigits = 3)
cols = [:koi_period, :koi_duration, :koi_depth, :koi_model_snr, :lc_snr, :lc_ntransits, :lc_sev_ratio, :lc_timing,
        :lc_vshape, :lc_depth_ratio, :lc_dur_ratio]
rows = [(feature = string(c), planets = q(pool[!, c], pl), nt_caught = q(pool[!, c], nt .& .!missed),
         nt_missed = q(pool[!, c], missed)) for c in cols]
println("medians:"); show_df(DataFrame(rows))

# 3. missed NT FPs by period and by number of observed transits
bins(v, e) = [findlast(<=(x), e) for x in v]
for (name, v, e) in (("period [d]", pool.koi_period, [0, 1.5, 10, 50, 200, Inf]),
                     ("observed transits", coalesce.(pool.lc_ntransits, NaN), [0, 3, 6, 20, 100, Inf]))
    b = bins(Float64.(v), e)
    rs = [(bin = "$(e[k])–$(e[k + 1])", nt = sum(nt .& (b .== k)), missed_pct = round(100mean(s[nt .& (b .== k)] .>= 0.5); digits = 1),
           planets = sum(pl .& (b .== k))) for k in 1:length(e) - 1 if sum(nt .& (b .== k)) > 0]
    println("NT-only FPs by $name:"); show_df(DataFrame(rs))
end
