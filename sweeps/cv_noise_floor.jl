# Step 1: noise floor of the baseline MLP.
#   (A) repeated star-grouped 5-fold CV on the train+val pool: 5 fold shuffles x 10 model seeds = 250 models
#   (B) notebook-style runs on a single 80/10/10 split: 20 model seeds on split 42, and 20 split seeds
# The test split (split_by_star seed 42) is never used.
# Run headless:  julia sweeps/cv_noise_floor.jl
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

df = clean_koi(load_koi())
pool, _ = cv_pool(df)

# --- (A) repeated grouped CV ---------------------------------------------------
cv = cv_run(pool; k = 5, repeats = 5, seeds = 1:10, name = "baseline")
println("saved: ", save_cv("cv_baseline_r5s10.jld2", cv))

# --- (B) notebook-style single-split runs ---------------------------------------
# Same recipe as exo100926: train on 80%, early-stop on the 10% val, score the val.
rows = NamedTuple[]
# split seed 42 reproduces exo100926 exactly; other split seeds re-split ONLY the pool
# (8/9 train, 1/9 val by star, same sizes as 80/10) so sealed test KOIs never appear.
function run_split!(rows, kind, split_seed, model_seed)
    tr, va = split_seed == 42 ? split_by_star(df; seed = 42)[1:2] :
                                split_by_star(pool; at = (8 / 9,), seed = split_seed)
    data = prepare_data(tr, va, va)       # third slot unused; the test split never enters
    model, hist = train_model(data; seed = model_seed, verbose = false)
    p = Float64.(predict_proba(model, data.Xva))
    push!(rows, merge((kind = kind, split_seed = split_seed, model_seed = model_seed,
                       best_epoch = hist.best_epoch, n_val = nrow(va)), fold_metrics(va, p)))
end
for s in 1:20
    run_split!(rows, "model seed (split 42)", 42, s)
end
for ss in 101:120
    run_split!(rows, "split seed (model seed 1)", ss, 1)
end
single = DataFrame(rows)
jldsave(joinpath(CV_DIR, "single_split_runs.jld2"); single)
println(combine(groupby(single, :kind), :roc_auc => mean, :roc_auc => std, :roc_auc => minimum, :roc_auc => maximum,
                :roc_auc_conf => mean, :roc_auc_conf => std))
println("DONE")
