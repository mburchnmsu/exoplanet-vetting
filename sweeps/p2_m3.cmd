@echo off
rem Phase 2 M3: light-curve features, then the Phase 2a CV (stops if the features step fails).
cd /d "%~dp0.."
julia --project=. -t 2 sweeps/p2_features.jl > sweeps\results\p2_features.log 2>&1 || exit /b 1
julia --project=. -t 2 sweeps/p2_cv.jl > sweeps\results\p2_cv.log 2>&1
