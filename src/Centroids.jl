# ==============================================================================
# STEP 4, part 3: centroid features and supplemental-KOI scoping.
# Included automatically by VettingSetup.jl.
#
# Centroid features (user decision 10/13): allowed, but as a SEPARATE
# "catalog + centroid" model next to the catalog-only reference. They are
# measurements (where the transit light comes from), not verdicts. But the
# robovetter's CO flag thresholds these same numbers, so gains on CO false
# positives partly re-learn its rule (circularity).
#
# Missing values: filled with the TRAINING median inside prepare_data. There are
# deliberately no "was missing" indicator features: missingness is informative
# (17% planets vs 52%) and is total for KOIs with no TCE, so an indicator would
# smuggle in the provenance signal we keep out.
#
# Supplemental KOIs (user decision 10/13): keep them in training. Headline metrics
# on DR25-detected KOIs; report supplemental KOIs separately. Provenance is never a
# model input; `is_dr25` is for selecting rows and reporting only.
# ==============================================================================

const CENTROID_FEATURES = [
    :cent_dicco_msky,    # transit-source offset from the out-of-transit centroid [arcsec] (difference images)
    :cent_dicco_sig,     # ... in units of its uncertainty (the robovetter thresholds this at ~3σ)
    :cent_dikco_msky,    # transit-source offset from the KIC position of the target [arcsec]
    :cent_dikco_sig,     # ... in units of its uncertainty
    :cent_fwm_stat_sig,  # significance (p-value-like) of flux-weighted centroid motion
    :cent_fwm_soff,      # flux-weighted source offset, |(srao, sdeco)| [arcsec]
    :cent_fwm_poff,      # flux-weighted centroid shift in transit, |(prao, pdeco)| [arcsec]
]

_f(x) = ismissing(x) ? NaN : Float64(x)
_sig(v, e) = (isnan(v) || isnan(e) || e <= 0) ? NaN : v / e

"""
    add_centroid_features(df, df_raw) -> DataFrame

Copy of `df` (same rows, same order) with the CENTROID_FEATURES columns joined
from the raw table by kepoi_name. Missing values become NaN; prepare_data fills
them with the training median.
"""
function add_centroid_features(df::DataFrame, df_raw::DataFrame)
    cols = [:koi_dicco_msky, :koi_dicco_msky_err, :koi_dikco_msky, :koi_dikco_msky_err, :koi_fwm_stat_sig,
            :koi_fwm_srao, :koi_fwm_sdeco, :koi_fwm_prao, :koi_fwm_pdeco]
    j = leftjoin(df[:, [:kepoi_name]], df_raw[:, [:kepoi_name; cols]]; on = :kepoi_name, order = :left)
    @assert j.kepoi_name == df.kepoi_name
    out = copy(df)
    out.cent_dicco_msky   = _f.(j.koi_dicco_msky)
    out.cent_dicco_sig    = _sig.(_f.(j.koi_dicco_msky), _f.(j.koi_dicco_msky_err))
    out.cent_dikco_msky   = _f.(j.koi_dikco_msky)
    out.cent_dikco_sig    = _sig.(_f.(j.koi_dikco_msky), _f.(j.koi_dikco_msky_err))
    out.cent_fwm_stat_sig = _f.(j.koi_fwm_stat_sig)
    out.cent_fwm_soff     = hypot.(_f.(j.koi_fwm_srao), _f.(j.koi_fwm_sdeco))
    out.cent_fwm_poff     = hypot.(_f.(j.koi_fwm_prao), _f.(j.koi_fwm_pdeco))
    return out
end

"Bool per row of `df`: detected by the final DR25 pipeline (for row selection and reporting only, never a feature)."
function is_dr25(df::DataFrame, df_raw::DataFrame)
    j = leftjoin(df[:, [:kepoi_name]], df_raw[:, [:kepoi_name, :koi_tce_delivname]]; on = :kepoi_name, order = :left)
    @assert j.kepoi_name == df.kepoi_name
    return coalesce.(j.koi_tce_delivname .== "q1_q17_dr25_tce", false)
end
