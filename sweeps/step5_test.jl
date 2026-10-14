# ONE-TIME sealed-test evaluation (10/13/2026, opened with the user's go-ahead).
# Scores the test split of split_by_star(seed 42) with the two frozen models from models/step5_frozen.jld2
# and saves the predictions. Nothing is fitted or tuned here. The notebook (exo101326 section 12) reads the CSV.
# Refuses to run again if the predictions already exist, so the test set is only scored once.
#   julia sweeps/step5_test.jl
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

out = joinpath(CV_DIR, "test_predictions.csv")
isfile(out) && error("test predictions already exist ($out): the sealed test set is scored only once")

df_raw = load_koi()
df = clean_koi(df_raw)
pool, test0 = cv_pool(df)
@assert isempty(intersect(Set(pool.kepid), Set(test0.kepid)))     # star-disjoint from everything trained on
test = add_centroid_features(test0, df_raw)
frozen = JLD2.load(joinpath(PROJECT_DIR, "models", "step5_frozen.jld2"))["frozen"]
pred = DataFrame(kepoi_name = test.kepoi_name, kepid = test.kepid, koi_disposition = test.koi_disposition,
                 label = test.label, dr25 = is_dr25(test, df_raw),
                 p_catalog = frozen_predict(frozen["catalog_only"], test),
                 p_centroid = frozen_predict(frozen["catalog_centroid"], test))
ks = Dict(zip(df_raw.kepoi_name, df_raw.koi_score))
pred.koi_score = [coalesce(get(ks, k, missing), NaN) for k in pred.kepoi_name]
CSV.write(out, pred)
@printf("scored %d test KOIs (%d stars; %d DR25) -> %s\n", nrow(pred), length(unique(pred.kepid)), sum(pred.dr25), out)
