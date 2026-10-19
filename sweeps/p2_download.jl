# Phase 2, M1: download every star of the cleaned catalog (pool + test) into the compact light-curve cache.
# Resumable: stars already "ok" in data/lightcurves/manifest.csv are skipped, so just rerun after a kill.
# No labels are used. Test stars are downloaded by the same code; nothing about their labels is looked at.
#   cmd /c "start /b /wait /belownormal julia --project=. -t 2 sweeps/p2_download.jl"
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

df = clean_koi(load_koi())
stars = sort(unique(df.kepid))
download_stars(stars; ntasks = 4, log_every = 100)

man = CSV.read(MANIFEST, DataFrame)
last_row = combine(groupby(man, :kepid), last)            # a retried star keeps its latest status
ok = filter(r -> r.kepid in Set(stars), last_row)
@printf("\nM1: %d / %d stars ok, %d errors; cache %.1f GB\n", count(==("ok"), ok.status), length(stars),
        count(!=("ok"), ok.status), sum(ok.cache_bytes) / 1e9)
