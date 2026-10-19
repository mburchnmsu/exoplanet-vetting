# ==============================================================================
# LIGHT CURVES (Phase 2): MAST listing/download of Kepler DR25 long-cadence
# light curves, FITS reading, and the compact per-star cache.
#
# Streaming per star (PHASE2_PLAN D2): download the star's _llc.fits files to a
# temporary folder, keep only the columns we need, write one JLD2 per star to
# data/lightcurves/cache/, delete the FITS. Nothing here looks at labels.
# ==============================================================================
using FITSIO, Dates

const LC_DIR    = joinpath(DATA_DIR, "lightcurves")       # git-ignored (D7)
const CACHE_DIR = joinpath(LC_DIR, "cache")
const MANIFEST  = joinpath(LC_DIR, "manifest.csv")
const MAST_LC   = "https://archive.stsci.edu/missions/kepler/lightcurves"

kic9(kepid) = lpad(string(kepid), 9, '0')
mast_star_url(kepid) = string(MAST_LC, "/", kic9(kepid)[1:4], "/", kic9(kepid), "/")
cache_path(kepid) = joinpath(CACHE_DIR, "kplr$(kic9(kepid)).jld2")

"Download `url` to `dest` with retries and exponential back-off."
function fetch_retry(url, dest; tries = 5, timeout = 120)
    for k in 1:tries
        try
            return Downloads.download(url, dest; timeout)
        catch e
            k == tries && rethrow()
            sleep(2.0^k)
        end
    end
end

"File names of the star's long-cadence light curves (one per quarter, Q0–Q17)."
function list_llc(kepid)
    io = IOBuffer()
    fetch_retry(mast_star_url(kepid), io)
    html = String(take!(io))
    return sort(unique(m.match for m in eachmatch(r"kplr\d{9}-\d{13}_llc\.fits", html)))
end

"Read one _llc.fits file: the columns Phase 2 uses, finite-time rows only."
function read_llc(path)
    FITS(path) do f
        q = read_header(f[1])["QUARTER"]
        h = f[2]
        t = read(h, "TIME")
        keep = isfinite.(t)
        col(name, T) = T.(read(h, name)[keep])
        return (time = t[keep], flux = col("PDCSAP_FLUX", Float32), flux_err = col("PDCSAP_FLUX_ERR", Float32),
                quality = col("SAP_QUALITY", Int32), cen1 = col("MOM_CENTR1", Float32),
                cen2 = col("MOM_CENTR2", Float32), quarter = fill(Int8(q), count(keep)))
    end
end

"""
    cache_star(kepid; ntasks = 4) -> (nfiles, bytes)

Download every quarter of one star (`ntasks` parallel requests), reduce to the
compact cache file and delete the raw FITS. Skips the star if its cache exists.
"""
function cache_star(kepid; ntasks = 4, tmpdir = joinpath(LC_DIR, "tmp"))
    isfile(cache_path(kepid)) && return (0, 0)
    files = list_llc(kepid)
    isempty(files) && error("no long-cadence files on MAST")
    dir = mkpath(joinpath(tmpdir, kic9(kepid)))
    try
        paths = asyncmap(fn -> fetch_retry(mast_star_url(kepid) * fn, joinpath(dir, fn)), files; ntasks)
        bytes = sum(filesize, paths)
        parts = [read_llc(p) for p in paths]
        lc = (; (k => reduce(vcat, getfield.(parts, k)) for k in keys(parts[1]))...)
        o = sortperm(lc.time)
        mkpath(CACHE_DIR)
        tmp = cache_path(kepid) * ".part"
        jldsave(tmp; kepid, files, (k => getfield(lc, k)[o] for k in keys(lc))...)
        mv(tmp, cache_path(kepid); force = true)                 # atomic: no half-written caches
        return (length(files), bytes)
    finally
        rm(dir; recursive = true, force = true)
    end
end

"Load a star's cached light curve as a NamedTuple (time, flux, flux_err, quality, cen1, cen2, quarter)."
function load_lc(kepid)
    d = JLD2.load(cache_path(kepid))
    return (time = d["time"], flux = d["flux"], flux_err = d["flux_err"], quality = d["quality"],
            cen1 = d["cen1"], cen2 = d["cen2"], quarter = d["quarter"])
end

"Stars already in the manifest with status \"ok\"."
done_stars() = isfile(MANIFEST) ?
    Set(r.kepid for r in CSV.File(MANIFEST) if r.status == "ok" && isfile(cache_path(r.kepid))) : Set{Int}()

"""
    download_stars(kepids; ntasks = 4)

Resumable loop: cache each star not yet done and append a manifest row
(kepid, nfiles, raw_bytes, cache_bytes, seconds, status, error, finished).
"""
function download_stars(kepids; ntasks = 4, log_every = 25)
    mkpath(LC_DIR)
    isfile(MANIFEST) || write(MANIFEST, "kepid,nfiles,raw_bytes,cache_bytes,seconds,status,error,finished\n")
    done = done_stars()
    todo = [k for k in kepids if !(k in done)]
    @info "download_stars" requested = length(kepids) already_done = length(kepids) - length(todo)
    t_start = time()
    for (i, k) in enumerate(todo)
        t0 = time()
        row = try
            nf, b = cache_star(k; ntasks)
            (nf, b, filesize(cache_path(k)), time() - t0, "ok", "")
        catch e
            msg = replace(sprint(showerror, e), r"[\r\n,\"]" => " ")
            (0, 0, 0, time() - t0, "error", first(msg, 200))
        end
        open(MANIFEST, "a") do io
            @printf(io, "%d,%d,%d,%d,%.1f,%s,\"%s\",%s\n", k, row..., Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
        end
        if i % log_every == 0 || i == length(todo)
            el = (time() - t_start) / 3600
            @printf("%s  %d/%d stars, %.2f h elapsed, ~%.1f h left\n", Dates.format(now(), "HH:MM"), i,
                    length(todo), el, el / i * (length(todo) - i))
            flush(stdout)
        end
    end
end
