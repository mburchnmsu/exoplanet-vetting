# ==============================================================================
# CNN (Phase 2b): AstroNet-style two-tower 1D CNN on the light-curve views,
# optionally joined with the (quantile-transformed) tabular features before the
# dense head. Sized for a CPU: ~200k parameters.
#   global tower: 4 × [Conv1D(k=5) ×2 → MaxPool(4)], 8 → 64 channels (2001 → 7 bins)
#   local tower : 2 × [Conv1D(k=5) ×2 → MaxPool(2)], 16 → 32 channels (201 → 50 bins);
#                 input channels from `local_keys` (local, odd, even, secondary, centroids)
# Plugs into cv_run via `trainer = train_cnn, extra = view_extra(V, keys)`.
# ==============================================================================

struct CNNVetter
    g::Chain
    l::Chain
    head::Chain
    use_feats::Bool
end
Flux.@layer CNNVetter

conv_block(cin, cout, pool) = Chain(Conv((5,), cin => cout, relu; pad = SamePad()),
                                    Conv((5,), cout => cout, relu; pad = SamePad()), MaxPool((pool,)))

function (m::CNNVetter)(G, L, X)
    z = vcat(Flux.flatten(m.g(G)), Flux.flatten(m.l(L)))
    return m.head(m.use_feats ? vcat(z, X) : z)
end

function build_cnn(n_local_ch, n_feats; dropout = 0.3, width = 64)
    g = Chain(conv_block(1, 8, 4), conv_block(8, 16, 4), conv_block(16, 32, 4), conv_block(32, 64, 4))
    l = Chain(conv_block(n_local_ch, 16, 2), conv_block(16, 32, 2))
    nz = size(g(zeros(Float32, N_GLOBAL, 1, 1)), 1) * 64 + size(l(zeros(Float32, N_LOCAL, n_local_ch, 1)), 1) * 32
    head = Chain(Dense(nz + n_feats => width, relu), Dropout(dropout), Dense(width => 1))
    return CNNVetter(g, l, head, n_feats > 0)
end

n_params(m) = sum(length, Flux.trainables(m))

"""
    view_arrays(V, names, local_keys; cen_scale) -> (G, L)

Global view [2001, 1, n] and stacked local-type views [201, n_ch, n] for KOIs `names`, from the views file `V`
(`data/views_all.jld2`). Flux views are clamped to ±5 (shallow, noisy KOIs); centroid views are divided by
`cen_scale` (a fixed robust scale) and clamped to ±10.
"""
function view_arrays(V, names, local_keys; cen_scale = 1f-3)
    col = V["__col"]
    idx = [col[n] for n in names]
    G = reshape(clamp.(V["global_view"][:, idx], -5f0, 5f0), N_GLOBAL, 1, :)
    L = Array{Float32}(undef, N_LOCAL, length(local_keys), length(idx))
    for (c, k) in enumerate(local_keys)
        L[:, c, :] = startswith(String(k), "cen") ? clamp.(V[String(k)][:, idx] ./ cen_scale, -10f0, 10f0) :
                     clamp.(V[String(k)][:, idx], -5f0, 5f0)
    end
    return G, L
end

"Load the views file once, with a name -> column lookup."
function load_views()
    V = JLD2.load(joinpath(DATA_DIR, "views_all.jld2"))
    V["__col"] = Dict(n => i for (i, n) in enumerate(V["names"]))
    return V
end

"An `extra` function for cv_run: adds the views of each fold part to its data."
view_extra(V, local_keys; kw...) = (tr, va, te) -> begin
    Gtr, Ltr = view_arrays(V, tr.kepoi_name, local_keys; kw...)
    Gva, Lva = view_arrays(V, va.kepoi_name, local_keys; kw...)
    Gte, Lte = view_arrays(V, te.kepoi_name, local_keys; kw...)
    (Gtr = Gtr, Ltr = Ltr, Gva = Gva, Lva = Lva, Gte = Gte, Lte = Lte)
end

function cnn_predict(m, G, L, X; batch = 512)
    Flux.testmode!(m)
    n = size(G, 3)
    out = Vector{Float32}(undef, n)
    for a in 1:batch:n
        r = a:min(a + batch - 1, n)
        out[r] = σ.(vec(m(G[:, :, r], L[:, :, r], X[:, r])))
    end
    return out
end

"Trained CNN plus nothing else: predictions need the fold's views, so cv_run calls predict_test."
predict_test(m::CNNVetter, data) = cnn_predict(m, data.Gte, data.Lte, data.Xte)

"""
    train_cnn(data; seed, epochs = 60, patience = 8, lr = 1e-3, wd = 1e-4, batchsize = 64, use_feats = true,
              dropout = 0.3, mirror = true, max_seconds = Inf, verbose = false)

AdamW, mini-batches, early stopping on the inner validation loss (best weights kept), as in `train_model`.
`mirror`: each batch is time-reversed with probability 1/2 (transits are symmetric to first order).
Returns `(model, hist)` with `hist.best_epoch` and `hist.epoch_seconds`.
"""
function train_cnn(data; seed = 1, epochs = 60, patience = 8, lr = 1e-3, wd = 1e-4, batchsize = 64,
                   use_feats = true, dropout = 0.3, mirror = true, max_seconds = Inf, verbose = false)
    Random.seed!(seed); rng = Xoshiro(seed)
    nf = use_feats ? size(data.Xtr, 1) : 0
    m = build_cnn(size(data.Ltr, 2), nf; dropout)
    opt = Flux.setup(Optimisers.AdamW(lr, (0.9, 0.999), wd), m)
    Xtr = use_feats ? data.Xtr : zeros(Float32, 0, size(data.Xtr, 2))
    Xva = use_feats ? data.Xva : zeros(Float32, 0, size(data.Xva, 2))
    ytr, yva = data.ytr, data.yva
    n = length(ytr)
    lossf(mm, G, L, X, y) = Flux.logitbinarycrossentropy(vec(mm(G, L, X)), y)
    val_loss() = (p = clamp.(Float64.(cnn_predict(m, data.Gva, data.Lva, Xva)), 1e-7, 1 - 1e-7);
                  -mean(@. yva * log(p) + (1 - yva) * log(1 - p)))
    hist = (val = Float64[], epoch_seconds = Float64[])
    best, best_state, best_epoch = Inf, deepcopy(Flux.state(m)), 0
    t_start = time()
    for epoch in 1:epochs
        t0 = time()
        Flux.trainmode!(m)
        for b in Iterators.partition(randperm(rng, n), batchsize)
            G, L, X, y = data.Gtr[:, :, b], data.Ltr[:, :, b], Xtr[:, b], ytr[b]
            if mirror && rand(rng) < 0.5
                G = reverse(G; dims = 1); L = reverse(L; dims = 1)
            end
            g = Flux.gradient(mm -> lossf(mm, G, L, X, y), m)
            Flux.update!(opt, m, g[1])
        end
        push!(hist.val, val_loss()); push!(hist.epoch_seconds, time() - t0)
        verbose && @printf("epoch %3d  val %.4f  (%.1f s)\n", epoch, hist.val[end], hist.epoch_seconds[end])
        if hist.val[end] < best
            best, best_state, best_epoch = hist.val[end], deepcopy(Flux.state(m)), epoch
        elseif epoch - best_epoch >= patience
            break
        end
        time() - t_start > max_seconds && break
    end
    Flux.loadmodel!(m, best_state)
    Flux.testmode!(m)
    m = use_feats ? m : CNNNoFeats(m)
    return m, merge(hist, (best_epoch = best_epoch,))
end

"Wrapper so a views-only CNN ignores the tabular features at prediction time."
struct CNNNoFeats
    m::CNNVetter
end
predict_test(w::CNNNoFeats, data) = cnn_predict(w.m, data.Gte, data.Lte, zeros(Float32, 0, size(data.Gte, 3)))
