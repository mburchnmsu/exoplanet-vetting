# Phase 4, M2 follow-up (pool only, analysis): why did N-a/N-b/N-c not reduce NT misses? Single-feature AUCs of
# NT-only FPs vs planets INSIDE SNR bins (koi_model_snr), where the missed NT FPs live. A feature that only
# separates NT from planets through SNR adds nothing over lc_snr / koi_model_snr, which the models already have.
#   julia --project=. -t 2 sweeps/p4_snrmatched.jl
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

df_raw = load_koi()
pool, _ = cv_pool(clean_koi(df_raw))
pool = leftjoin(pool, CSV.read(joinpath(CV_DIR, "p2_lcfeatures.csv"), DataFrame); on = :kepoi_name, order = :left)
pool = leftjoin(pool, CSV.read(joinpath(CV_DIR, "p4_badtimes.csv"), DataFrame)[:, [:kepoi_name; BADTIME_FEATURES]];
                on = :kepoi_name, order = :left)
pool = leftjoin(pool, CSV.read(joinpath(CV_DIR, "p4_ntfeatures.csv"), DataFrame); on = :kepoi_name, order = :left)
const NDFILE = joinpath(CV_DIR, "p4_ndfeatures.csv")
isfile(NDFILE) && (pool = leftjoin(pool, CSV.read(NDFILE, DataFrame); on = :kepoi_name, order = :left))
y = pool.label
nt = (y .== 0) .& (with_vetting_info(pool, df_raw, zeros(nrow(pool))).flag_pattern .== "NT")
snr = Float64.(coalesce.(pool.koi_model_snr, NaN))
show_df(t) = (show(stdout, MIME"text/plain"(), t; allrows = true, allcols = true); println(); flush(stdout))
bins = [(7.0, 10.0), (10.0, 15.0), (15.0, 30.0)]
rows = NamedTuple[]
for c in [:lc_snr, :lc_vshape, :lc_sev_ratio, :lc_timing, :lc_sec_sig, BADTIME_FEATURES..., LC_NT_FEATURES...,
          (isfile(NDFILE) ? LC_ND_FEATURES : Symbol[])...]
    x = Float64.(coalesce.(pool[!, c], NaN))
    r = Dict{Symbol, Any}(:feature => string(c))
    for (lo, hi) in bins
        m = (lo .<= snr .< hi) .& ((y .== 1) .| nt) .& isfinite.(x)
        a = fast_auc(Float64.(y[m]), x[m])
        r[Symbol("auc_snr_$(Int(lo))_$(Int(hi))")] = round(max(a, 1 - a); digits = 3)
    end
    push!(rows, NamedTuple(r))
end
@printf("NT-only FPs / planets per koi_model_snr bin: %s\n",
        join(["[$(lo), $(hi)): $(sum(nt .& (lo .<= snr .< hi))) / $(sum((y .== 1) .& (lo .<= snr .< hi)))" for (lo, hi) in bins], ", "))
println("single-feature AUC, NT-only FPs vs planets within SNR bins (folded: 0.5 useless):")
show_df(select(DataFrame(rows), :feature, :auc_snr_7_10, :auc_snr_10_15, :auc_snr_15_30))
