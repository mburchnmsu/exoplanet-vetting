# Exoplanet vetting — Kepler KOI planet vs false-positive classifier

Julia/Flux project classifying Kepler cumulative KOIs (CONFIRMED+CANDIDATE vs FALSE POSITIVE) from catalog features.

- Code: `src/` (VettingSetup.jl and friends). Experiments: `sweeps/` (scripts, `results/`, `nbcells/`).
  Dated notebooks at the root (`exoMMDDYY.ipynb`). Jupyter isn't required: `sweeps/tools/run_nb.jl` runs notebooks headless.
- Read `sweeps/REPORT.md` first. Reuse its cross-validation protocol for any comparison.
- The held-out test split stays sealed; CV runs on the pool.
- Ask the user before acting on open design decisions (see the report).

## Syncing between PCs (GitHub: mburchnmsu/exoplanet-vetting, private)

The user works on two PCs. Keep the repo in sync without being asked:
- Start of a session: `git pull --ff-only` (a hook may already have done it).
- End of a session, or when the user signs off: commit the session's changes with a clear message and `git push`.
- If a pull or push conflicts, stop and tell the user. Don't force-push or resolve it silently.

Keep this project separate from the zodi emulator project: different repos, different sessions.

## Notify the user on major progress

The user steers sessions from their phone and wants a push notification (PushNotification tool) when something big
happens, even if they didn't ask in that session. Major means: a sweep or long run finishes or crashes, a new best model
or a result that changes the plan, a notebook finished for the mentor, or a blocker that needs their decision.
Not routine steps. One line, lead with the result, e.g. "Round 9 done: parallax bands fix ecliptic stripe, 4x worse elsewhere".