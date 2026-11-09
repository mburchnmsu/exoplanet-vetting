# Phase 3, M0: the small catalogs for ephemeris matching -> data/catalogs/ (git-ignored, a few MB).
# Only ephemeris, shape, position and brightness columns are downloaded (PHASE3_PLAN.md, E4): no dispositions,
# robovetter flags or scores of any object.
#   julia --project=. sweeps/p3_download.jl
using Downloads

const DIR = joinpath(@__DIR__, "..", "data", "catalogs")
const TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query="
mkpath(DIR)
for (file, adql) in [("dr25_tce.csv", "select kepid,tce_plnt_num,tce_period,tce_time0bk,tce_duration,tce_depth," *
                                      "tce_model_snr,ra,dec from q1_q17_dr25_tce"),
                     ("dr25_stellar.csv", "select kepid,kepmag,ra,dec from q1_q17_dr25_ks")]
    path = joinpath(DIR, file)
    Downloads.download(TAP * replace(adql, " " => "+", "," => "%2C"), path)
    println(file, ": ", countlines(path) - 1, " rows")
end
# Kepler EB catalog, third revision (Kirk et al. 2016, VizieR J/AJ/151/68/catalog): KIC, period, epoch
# (BJD − 2400000), Kepler magnitude. No depth or duration in the table. CDS and Villanova were unreachable on
# 11/04/2026; the CfA VizieR mirror works (11/08). The Morph column and the paper's sub-tables are not used.
path = joinpath(DIR, "kepler_eb.tsv")
Downloads.download("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=J/AJ/151/68/catalog&-out.max=unlimited" *
                   "&-out=KIC&-out=Per&-out=BJD0&-out=Kpmag", path)
println("kepler_eb.tsv: ", count(l -> !isempty(l) && isdigit(lstrip(l)[1]), eachline(path)), " data rows")
