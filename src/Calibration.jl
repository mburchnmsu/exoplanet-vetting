# ==============================================================================
# CALIBRATION AND OPERATING POINTS (step 5).
#  - Per-repeat scores: a deployable model is ONE 10-seed ensemble per fold
#    shuffle, so calibration uses the seed-mean of each repeat separately (the
#    repeat-mean in oof_scores is a bigger ensemble than anything we ship).
#  - Calibrators (monotone maps score -> probability): :none, :platt (logistic
#    on logit(score)), :isotonic (pool-adjacent-violators, linear interpolation).
#  - Everything is nested: for held-out fold f, the calibrator and the threshold
#    are fit on the OTHER folds' out-of-fold scores, so the reported numbers are
#    honest estimates for an unseen star.
# Included automatically by VettingSetup.jl.
# ==============================================================================

"Score new KOIs with a frozen model from models/step5_frozen.jld2 (50/50 average of the seed-mean MLP and trees)."
function frozen_predict(m, df::DataFrame)
    X = transform_new(m["prep"], df)
    p_mlp = mean(predict_proba(Flux.loadmodel!(build_mlp(size(X, 1)), st), X) for st in m["mlp_states"])
    p_tree = mean(predict_proba(t, X) for t in m["trees"])
    return Float64.((p_mlp .+ p_tree) ./ 2)
end

"[n_pool, repeats] seed-mean OOF score of each fold shuffle."
repeat_scores(cv) = dropdims(mean(cv.oof; dims = 3); dims = 3)

const P_EPS = 1e-6
pclip(p) = clamp(p, P_EPS, 1 - P_EPS)
plogit(p) = log(pclip(p) / (1 - pclip(p)))

"Brier score (mean squared error of probabilities)."
brier(y, p) = mean(@. (p - y)^2)

"Expected calibration error over `bins` equal-count bins (|mean p - planet fraction|, count-weighted)."
function ece(y, p; bins = 15)
    o = sortperm(p)
    edges = round.(Int, range(0, length(p); length = bins + 1))
    return sum(abs(mean(p[o[edges[i]+1:edges[i+1]]]) - mean(y[o[edges[i]+1:edges[i+1]]])) * (edges[i+1] - edges[i])
               for i in 1:bins) / length(p)
end

# --- calibrators -------------------------------------------------------------------
"Platt scaling: P(planet) = σ(a·logit(s) + b), fit by Newton's method (2 parameters, convex)."
function fit_platt(s, y; iters = 50)
    z = plogit.(s)
    a, b = 1.0, 0.0
    for _ in 1:iters
        p = @. 1 / (1 + exp(-(a * z + b)))
        w = @. max(p * (1 - p), 1e-12)
        g = [sum(@. (p - y) * z), sum(p .- y)]
        H = [sum(w .* z .^ 2)+1e-9 sum(w .* z); sum(w .* z) sum(w)+1e-9]
        step = H \ g
        a, b = a - step[1], b - step[2]
        maximum(abs, step) < 1e-10 && break
    end
    return (method = :platt, a = a, b = b)
end

"Isotonic regression (pool adjacent violators) of y on s; knots at block mean scores."
function fit_isotonic(s, y)
    o = sortperm(s)
    xs, ys, ws = Float64[], Float64[], Float64[]
    for i in o
        push!(xs, s[i]); push!(ys, y[i]); push!(ws, 1.0)
        while length(ys) > 1 && ys[end-1] >= ys[end]          # merge violating (or tied) blocks
            w = ws[end-1] + ws[end]
            xs[end-1] = (ws[end-1] * xs[end-1] + ws[end] * xs[end]) / w
            ys[end-1] = (ws[end-1] * ys[end-1] + ws[end] * ys[end]) / w
            ws[end-1] = w
            pop!(xs); pop!(ys); pop!(ws)
        end
    end
    return (method = :isotonic, x = xs, y = ys)
end

fit_calibrator(method, s, y) =
    method == :none ? (method = :none,) : method == :platt ? fit_platt(s, y) :
    method == :isotonic ? fit_isotonic(s, y) : error("unknown calibrator $method")

"Apply a fitted calibrator to scores (isotonic: linear between knots, flat outside)."
function calibrate(c, s::AbstractVector)
    c.method == :none && return Float64.(s)
    c.method == :platt && return @. 1 / (1 + exp(-(c.a * plogit(s) + c.b)))
    x, v = c.x, c.y
    out = similar(s, Float64)
    for (k, t) in enumerate(s)
        j = searchsortedlast(x, t)
        out[k] = j == 0 ? v[1] : j == length(x) ? v[end] :
                 v[j] + (v[j+1] - v[j]) * (t - x[j]) / (x[j+1] - x[j])
    end
    return out
end

"""
    nested_calibrate(S, folds, y, method) -> [n, repeats]

For every repeat r and fold f: fit the calibrator on the other folds' scores
S[:, r] and apply it to fold f.
"""
function nested_calibrate(S, folds, y, method)
    P = similar(S, Float64)
    for r in axes(S, 2), f in unique(folds[:, r])
        held = folds[:, r] .== f
        c = fit_calibrator(method, S[.!held, r], y[.!held])
        P[held, r] = calibrate(c, S[held, r])
    end
    return P
end

# --- operating points --------------------------------------------------------------
"""
    pick_threshold(y, p, rule) -> t     (call planet if p >= t)

rule = (:recall, x)    highest t keeping at least a fraction x of planets
       (:precision, x) lowest t whose planet list is at least x pure
       (:maxf1,)       t maximizing F1
       (:prob, x)      fixed t = x
"""
function pick_threshold(y, p, rule)
    rule[1] == :prob && return rule[2]
    o = sortperm(p; rev = true)
    tp = cumsum(y[o] .== 1); k = 1:length(o)
    n1 = sum(y .== 1)
    # only cut between distinct scores: index k is valid if p[o[k]] > p[o[k+1]]
    ok = [i == length(o) || p[o[i]] > p[o[i+1]] for i in k]
    if rule[1] == :recall
        i = findfirst(i -> ok[i] && tp[i] >= rule[2] * n1, k)
    elseif rule[1] == :precision
        i = findlast(i -> ok[i] && tp[i] / i >= rule[2], k)
    elseif rule[1] == :maxf1
        f1 = [ok[i] ? 2tp[i] / (i + n1) : -1.0 for i in k]
        i = argmax(f1)
    else
        error("unknown rule $rule")
    end
    i === nothing && return NaN
    return p[o[i]]
end

"Confusion counts and rates for `p >= t` (positive = planet)."
function op_metrics(y, p, t)
    pred = p .>= t
    tp, fp = sum(pred .& (y .== 1)), sum(pred .& (y .== 0))
    fn, tn = sum(.!pred .& (y .== 1)), sum(.!pred .& (y .== 0))
    return (tp = tp, fp = fp, fn = fn, tn = tn, recall = tp / max(tp + fn, 1),
            precision = tp / max(tp + fp, 1), fp_kept = fp / max(fp + tn, 1))
end

"""
    nested_calls(P, folds, y, select, rule) -> (calls = Bool[n, repeats], t = thresholds[repeat, fold])

Honest hard calls: for every repeat and fold, the threshold is chosen with `rule`
on the other folds' KOIs that are in `select` (e.g. DR25 KOIs) and applied to
every KOI of the held-out fold.
"""
function nested_calls(P, folds, y, select, rule)
    calls = falses(size(P))
    k = maximum(folds)
    T = fill(NaN, size(P, 2), k)
    for r in axes(P, 2), f in 1:k
        held = folds[:, r] .== f
        fit = .!held .& select
        T[r, f] = pick_threshold(y[fit], P[fit, r], rule)
        calls[held, r] = P[held, r] .>= T[r, f]
    end
    return (calls = calls, t = T)
end
