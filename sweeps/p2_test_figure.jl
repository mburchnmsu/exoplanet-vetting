# Phase 2, M6 figure: sealed-test ROC curves on DR25 KOIs, Phase 1 vs Phase 2 (both families), with each model's
# frozen 98%-recall operating point. Reads only the saved test predictions. -> figures/p2_test_roc.png
#   julia --project=. sweeps/p2_test_figure.jl
include(joinpath(@__DIR__, "..", "src", "VettingSetup.jl"))

p2 = CSV.read(joinpath(CV_DIR, "p2_test_predictions.csv"), DataFrame)
p1 = CSV.read(joinpath(CV_DIR, "test_predictions.csv"), DataFrame)
@assert p1.kepoi_name == p2.kepoi_name
f1 = JLD2.load(joinpath(PROJECT_DIR, "models", "step5_frozen.jld2"))["frozen"]
f2 = JLD2.load(joinpath(PROJECT_DIR, "models", "p2_frozen.jld2"))["frozen"]
drt = Bool.(p2.dr25)
y = Int.(p2.label[drt])
fig = Figure(size = (980, 470))
for (j, (ttl, key, a, b)) in enumerate((("Catalog-only", "catalog_only", p1.p_catalog, p2.p_catalog),
                                        ("Catalog + centroid", "catalog_centroid", p1.p_centroid, p2.p_centroid)))
    ax = Axis(fig[1, j]; xlabel = "false-positive rate (FPs kept)", ylabel = j == 1 ? "planet recall" : "",
              title = "$ttl, DR25 test KOIs", aspect = 1, limits = (0, 0.5, 0.75, 1.0))
    for (lab, p, f, col) in (("Phase 1 (catalog)", a, f1, :gray50), ("Phase 2 (+ light-curve features)", b, f2, :steelblue))
        fpr, tpr = roccurve(y, p[drt])
        lines!(ax, fpr, tpr; color = col, linewidth = 2, label = @sprintf("%s: AUC %.3f", lab, fast_auc(y, p[drt])))
        o = op_metrics(Float64.(y), p[drt], f[key]["thresholds"]["(:recall, 0.98)"])
        scatter!(ax, [o.fp_kept], [o.recall]; color = col, markersize = 12)
    end
    ks = Bool.(isfinite.(p1.koi_score[drt]))
    fpr, tpr = roccurve(y[ks], p1.koi_score[drt][ks])
    lines!(ax, fpr, tpr; color = :darkorange, linestyle = :dash,
           label = @sprintf("robovetter koi_score: AUC %.3f", fast_auc(Float64.(y[ks]), p1.koi_score[drt][ks])))
    axislegend(ax; position = :rb, labelsize = 11)
end
Label(fig[2, 1:2], "Dots: each model's frozen 98%-recall threshold. Zoomed to FP rate ≤ 0.5, recall ≥ 0.75.";
      fontsize = 12, color = :gray30)
println("saved: ", savefig("p2_test_roc.png", fig))
