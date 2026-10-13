# ==============================================================================
# VETTING SETUP: environment, KOI data, leakage guard, star-grouped split,
# scaling, loaders, model, loss, training loop, metrics, plots, save/load.
# Every notebook and sweep script should `include` this file so they all share
# exactly the same preprocessing.
#
#   include(joinpath(raw"C:\JuliaProjects\Exoplanet vetting", "src", "VettingSetup.jl"))
# ==============================================================================

# --- Environment -------------------------------------------------------------
const PROJECT_DIR = dirname(@__DIR__)          # src/ lives one level below the project root
const DATA_DIR    = joinpath(PROJECT_DIR, "data")
const FIG_DIR     = joinpath(PROJECT_DIR, "figures")
const MODEL_DIR   = joinpath(PROJECT_DIR, "models")

import Pkg
Pkg.activate(PROJECT_DIR; io = devnull)
Pkg.instantiate()

using Downloads, CSV, DataFrames, Statistics, StatsBase, Random, Printf
using MLUtils, Flux, Optimisers, EvalMetrics, CairoMakie, JLD2

# --- Data source ---------------------------------------------------------------
const TAP_URL  = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
const KOI_PATH = joinpath(DATA_DIR, "koi_cumulative.csv")

# --- Features -------------------------------------------------------------------
# koi_insol dropped for now (nearly redundant with koi_teq); revisit later.
# koi_count (number of KOIs on the star) carries the multiplicity prior:
# KOIs in multi-candidate systems are far more likely to be real planets.
const FEATURES = [
    :koi_period,     # orbital period [days]
    :koi_duration,   # transit duration [hrs]
    :koi_depth,      # transit depth [ppm]
    :koi_impact,     # impact parameter
    :koi_model_snr,  # transit-model SNR
    :koi_prad,       # planet radius [Earth radii]
    :koi_teq,        # equilibrium temperature [K]
    :koi_steff,      # stellar effective temperature [K]
    :koi_srad,       # stellar radius [Solar radii]
    :koi_slogg,      # stellar log g [cgs]
    :koi_count,      # KOIs on this star
]

# Multi-order features get log10(x + eps). eps is per-feature because koi_depth
# contains exact zeros (1 ppm floor) while period/prad are strictly positive.
const LOG_FEATURES = Dict(:koi_period => 1e-6, :koi_depth => 1.0, :koi_prad => 1e-3)

# Columns that encode the disposition (directly or via the vetting pipeline).
# None of these may ever be used as a model input.
const LEAKY_COLS = [
    :koi_fpflag_nt, :koi_fpflag_ss, :koi_fpflag_co, :koi_fpflag_ec,   # robovetter FP flags
    :koi_disposition, :koi_pdisposition,                               # the labels themselves
    :koi_score,                                                        # robovetter confidence
    :kepler_name,                                                      # only confirmed planets get one
    :koi_comment, :koi_disp_prov, :koi_vet_stat, :koi_vet_date,        # vetting metadata
]

"Throw if any requested feature is a known leakage column."
function check_no_leakage(features = FEATURES)
    bad = intersect(Symbol.(features), LEAKY_COLS)
    isempty(bad) || error("Leakage: these columns must not be model inputs: $(bad)")
    return true
end
check_no_leakage()

# --- Loading & cleaning ---------------------------------------------------------
"Download the cumulative KOI table to data/ unless it is already cached."
function fetch_koi(; force = false, path = KOI_PATH)
    if force || !isfile(path)
        mkpath(dirname(path))
        @info "Downloading Kepler cumulative KOI table" TAP_URL
        Downloads.download(TAP_URL, path)
    end
    return path
end

"Read the raw KOI table (archive header lines start with '#')."
load_koi(path = fetch_koi()) = CSV.read(path, DataFrame; comment = "#")

const LABELS = Dict("CONFIRMED" => 1, "CANDIDATE" => 1, "FALSE POSITIVE" => 0)

"""
    clean_koi(df_raw; features = FEATURES)

Keep rows with a known disposition and every model feature present, add
`:label` (1 = CONFIRMED/CANDIDATE, 0 = FALSE POSITIVE). Keeps `kepid`,
`kepoi_name` and `koi_disposition` for bookkeeping, not as features.
"""
function clean_koi(df_raw::DataFrame; features = FEATURES)
    check_no_leakage(features)
    cols = unique([:kepid, :kepoi_name, :koi_disposition, features...])
    df = dropmissing(df_raw[:, cols], [:koi_disposition; features])
    filter!(r -> haskey(LABELS, r.koi_disposition), df)
    df.label = [LABELS[d] for d in df.koi_disposition]
    @info "clean_koi" raw = nrow(df_raw) kept = nrow(df) stars = length(unique(df.kepid)) planets = sum(df.label) false_pos = sum(1 .- df.label)
    return df
end

# --- Star-grouped, stratified split ---------------------------------------------
"""
    split_by_star(df; at = (0.8, 0.1), seed = 42) -> (train, val, test)

Every KOI of a given star (`kepid`) lands in the same split, so shared stellar
parameters and sibling planets cannot leak across splits. Stars are stratified
by system type (all planets / all false positives / mixed) to keep the label
balance similar in each split.
"""
function split_by_star(df::DataFrame; at = (0.8, 0.1), seed = 42)
    stars = star_systems(df)
    rng = Xoshiro(seed)
    parts = splitobs(rng, collect(1:nrow(stars)); at, shuffle = true, stratified = stars.system)
    ids = [Set(stars.kepid[collect(p)]) for p in parts]
    splits = Tuple(df[in.(df.kepid, Ref(s)), :] for s in ids)
    @assert all(isempty(intersect(ids[i], ids[j])) for i in eachindex(ids) for j in eachindex(ids) if i < j)
    return splits
end

"One row per star with its system type: all planets / all false positives / mixed."
star_systems(df::DataFrame) =
    combine(groupby(df, :kepid), :label => (l -> all(==(1), l) ? "planet" :
                                                 all(==(0), l) ? "fp" : "mixed") => :system)

"One-line-per-split summary of sizes and planet fraction."
function describe_splits(train, val, test)
    for (name, d) in zip(("train", "val", "test"), (train, val, test))
        @printf("%-5s  %5d KOIs  %5d stars  planet frac %.3f\n",
                name, nrow(d), length(unique(d.kepid)), mean(d.label))
    end
end

# --- Features -> matrices, scaling ---------------------------------------------
"Feature matrix [n_features, n_obs] (Float32) with log transforms applied."
function feature_matrix(df::DataFrame; features = FEATURES)
    check_no_leakage(features)
    X = Matrix{Float64}(undef, length(features), nrow(df))
    for (i, f) in enumerate(features)
        x = Float64.(df[!, f])
        X[i, :] = haskey(LOG_FEATURES, f) ? log10.(x .+ LOG_FEATURES[f]) : x
    end
    return Float32.(X)
end

labels(df::DataFrame) = Float32.(df.label)

"Per-feature mean/std. Fit this on the TRAINING split only."
fit_scaler(X::AbstractMatrix) = (μ = mean(X; dims = 2), σ = std(X; dims = 2) .+ 1f-8)
apply_scaler(s, X::AbstractMatrix) = (X .- s.μ) ./ s.σ

"""
    prepare_data(train, val, test; features = FEATURES)

Build scaled matrices for all three splits, with the scaler fit on `train` only.
Returns a NamedTuple with Xtr/ytr/Xva/yva/Xte/yte and the scaler.
"""
function prepare_data(train, val, test; features = FEATURES)
    Xtr = feature_matrix(train; features)
    s = fit_scaler(Xtr)
    return (Xtr = apply_scaler(s, Xtr), ytr = labels(train),
            Xva = apply_scaler(s, feature_matrix(val; features)), yva = labels(val),
            Xte = apply_scaler(s, feature_matrix(test; features)), yte = labels(test),
            scaler = s, features = features)
end

make_loader(X, y; batchsize = 128, shuffle = true, rng = Xoshiro(0)) =
    Flux.DataLoader((X, y); batchsize, shuffle, rng)

# --- Model & loss ---------------------------------------------------------------
"""
MLP that outputs a raw logit (no sigmoid; the loss and predict apply it).
`hidden = ()` gives plain logistic regression: a single Dense(n_features => 1).
"""
function build_mlp(n_features; hidden = (64, 32), dropout = 0.2, act = relu)
    isempty(hidden) && return Chain(Dense(n_features => 1))
    return Chain(
        Dense(n_features => hidden[1], act),
        Dropout(dropout),
        Dense(hidden[1] => hidden[2], act),
        Dense(hidden[2] => 1),
    )
end

"""
    weighted_logitbce(logits, y; pos_weight = 1)

Numerically stable weighted BCE on logits. `pos_weight > 1` penalizes missed
planets (false negatives) more than false alarms.
"""
function weighted_logitbce(logits, y; pos_weight = 1f0)
    z = vec(logits)
    return mean(@. -(pos_weight * y * logσ(z) + (1 - y) * logσ(-z)))
end

"Planet probability for each column of X."
function predict_proba(model, X)
    Flux.testmode!(model)
    return σ.(vec(model(X)))
end

# --- Training ---------------------------------------------------------------------
"""
    train_model(data; epochs = 200, lr = 1e-3, batchsize = 128, pos_weight = 1,
                patience = 25, seed = 1, hidden = (64, 32), dropout = 0.2)

Adam + mini-batches, tracking train/val loss each epoch. Keeps the weights with
the best validation loss (early stopping after `patience` epochs without
improvement). Returns `(model, history)`.
"""
function train_model(data; epochs = 200, lr = 1e-3, batchsize = 128, pos_weight = 1f0,
                     patience = 25, seed = 1, hidden = (64, 32), dropout = 0.2, verbose = true)
    Random.seed!(seed)
    model = build_mlp(size(data.Xtr, 1); hidden, dropout)
    opt = Flux.setup(Optimisers.Adam(lr), model)
    loader = make_loader(data.Xtr, data.ytr; batchsize, rng = Xoshiro(seed))
    pw = Float32(pos_weight)
    lossf(m, X, y) = weighted_logitbce(m(X), y; pos_weight = pw)

    hist = (train = Float64[], val = Float64[])
    best_val, best_state, best_epoch = Inf, deepcopy(Flux.state(model)), 0
    for epoch in 1:epochs
        Flux.trainmode!(model)
        for (X, y) in loader
            g = Flux.gradient(m -> lossf(m, X, y), model)
            Flux.update!(opt, model, g[1])
        end
        Flux.testmode!(model)
        push!(hist.train, lossf(model, data.Xtr, data.ytr))
        push!(hist.val, lossf(model, data.Xva, data.yva))
        if hist.val[end] < best_val
            best_val, best_state, best_epoch = hist.val[end], deepcopy(Flux.state(model)), epoch
        elseif epoch - best_epoch >= patience
            verbose && @info "Early stop" epoch best_epoch
            break
        end
        verbose && (epoch % 20 == 0) && @printf("epoch %4d  train %.4f  val %.4f\n", epoch, hist.train[end], hist.val[end])
    end
    Flux.loadmodel!(model, best_state)
    Flux.testmode!(model)
    return model, merge(hist, (best_epoch = best_epoch,))
end

# --- Metrics -------------------------------------------------------------------
"""
    evaluate(y, p; threshold = 0.5)

ROC-AUC, PR-AUC and confusion-matrix metrics at `threshold` (EvalMetrics).
"""
function evaluate(y, p; threshold = 0.5)
    t = Int.(y)
    cm = ConfusionMatrix(t, p, threshold)
    return (roc_auc = au_roccurve(t, p), pr_auc = au_prcurve(t, p),
            accuracy = accuracy(cm), precision = EvalMetrics.precision(cm),
            recall = EvalMetrics.recall(cm), f1 = f1_score(cm),
            tp = cm.tp, fp = cm.fp, tn = cm.tn, fn = cm.fn)
end

"""
    evaluate_subsets(df, p; threshold = 0.5)

Metrics on the full split and on CONFIRMED-vs-FALSE POSITIVE only (drops the
noisier CANDIDATE labels), as a DataFrame.
"""
function evaluate_subsets(df::DataFrame, p; threshold = 0.5)
    rows = NamedTuple[]
    push!(rows, merge((subset = "all (CONF+CAND vs FP)", n = nrow(df)), evaluate(df.label, p; threshold)))
    m = df.koi_disposition .!= "CANDIDATE"
    push!(rows, merge((subset = "CONFIRMED vs FP", n = sum(m)), evaluate(df.label[m], p[m]; threshold)))
    return DataFrame(rows)
end

# --- Plots ---------------------------------------------------------------------
function plot_history(hist; title = "Training history")
    fig = Figure(size = (650, 400))
    ax = Axis(fig[1, 1]; xlabel = "epoch", ylabel = "weighted BCE", title)
    lines!(ax, hist.train; label = "train")
    lines!(ax, hist.val; label = "val")
    vlines!(ax, [hist.best_epoch]; color = :gray, linestyle = :dash, label = "best val")
    axislegend(ax)
    return fig
end

function plot_roc_pr(y, p; title = "")
    t = Int.(y)
    fpr, tpr = roccurve(t, p)
    rec, prec = prcurve(t, p)
    fig = Figure(size = (900, 420))
    a1 = Axis(fig[1, 1]; xlabel = "false positive rate", ylabel = "true positive rate",
              title = @sprintf("ROC  (AUC = %.3f)  %s", au_roccurve(t, p), title), aspect = 1)
    lines!(a1, fpr, tpr); ablines!(a1, 0, 1; color = :gray, linestyle = :dash)
    a2 = Axis(fig[1, 2]; xlabel = "recall", ylabel = "precision",
              title = @sprintf("PR  (AUC = %.3f)  %s", au_prcurve(t, p), title), aspect = 1)
    lines!(a2, rec, prec); hlines!(a2, [mean(t)]; color = :gray, linestyle = :dash)
    limits!(a1, 0, 1, 0, 1); limits!(a2, 0, 1, 0, 1)
    return fig
end

function plot_confusion(y, p; threshold = 0.5, title = "Confusion matrix")
    cm = ConfusionMatrix(Int.(y), p, threshold)
    M = [cm.tn cm.fp; cm.fn cm.tp]                     # rows = truth, cols = prediction
    fig = Figure(size = (450, 400))
    ax = Axis(fig[1, 1]; title = "$title (t = $threshold)", xlabel = "predicted", ylabel = "true",
              xticks = (1:2, ["FP", "planet"]), yticks = (1:2, ["FP", "planet"]), yreversed = true)
    heatmap!(ax, 1:2, 1:2, permutedims(M); colormap = :Blues)
    for i in 1:2, j in 1:2
        text!(ax, j, i; text = string(M[i, j]), align = (:center, :center), fontsize = 20)
    end
    return fig
end

savefig(name, fig) = (mkpath(FIG_DIR); save(joinpath(FIG_DIR, name), fig); joinpath(FIG_DIR, name))

# --- Save / load ---------------------------------------------------------------
"Save weights + scaler + feature list + config so the model is usable on its own."
function save_vetter(name, model, data; config = Dict())
    mkpath(MODEL_DIR)
    path = joinpath(MODEL_DIR, name)
    jldsave(path; model_state = Flux.state(model), scaler = data.scaler,
            features = data.features, config = config)
    return path
end

function load_vetter(path; hidden = (64, 32), dropout = 0.2)
    d = JLD2.load(path)
    model = build_mlp(length(d["features"]); hidden, dropout)
    Flux.loadmodel!(model, d["model_state"])
    Flux.testmode!(model)
    return model, d["scaler"], d["features"], d["config"]
end

include(joinpath(@__DIR__, "CrossValidation.jl"))
include(joinpath(@__DIR__, "ErrorAnalysis.jl"))
include(joinpath(@__DIR__, "Baselines.jl"))

println("VettingSetup loaded:$(length(FEATURES)) features, data at $(KOI_PATH)")
