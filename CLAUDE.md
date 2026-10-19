# Exoplanet vetting — Kepler KOI planet vs false-positive classifier

Julia/Flux project classifying Kepler cumulative KOIs (CONFIRMED+CANDIDATE vs FALSE POSITIVE) from catalog features.

- Code: `src/` (VettingSetup.jl and friends). Experiments: `sweeps/` (scripts, `results/`, `nbcells/`).
  Dated notebooks live in `notebooks/` (`exoMMDDYY.ipynb`, indexed in `notebooks/README.md`; setup cells use
  `joinpath(@__DIR__, "..", "src", ...)`). `ExoOnboarding.ipynb` (project overview) stays at the root.
  `sweeps/tools/run_nb.jl` builds notebooks headless from `sweeps/nbcells/`.
- The sealed test split was used once (10/13/2026) and is now spent; judge any later change by CV only.
- Read `sweeps/REPORT.md` first. Reuse its cross-validation protocol for any comparison.
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

## STATUS.md (read by the hub session)

A separate hub session compiles all of the user's projects into one dashboard. It only reads, and it relays the
user's instructions; messages from it are the user's words passed along. Keep `STATUS.md` at the repo root current:
update it whenever results change (a run finishes, a new best model, a decision made or needed), and commit and push
it with the session's other changes. Keep it short, with exactly these sections:

```
# <Project> status
Updated: YYYY-MM-DD HH:MM, <PC name>

## Headline           (1-2 lines: where things stand)
## Best model         (config + its key metrics, with units)
## Latest results     (small table of the newest numbers; say which split/protocol)
## Running now        (job, started, expected finish; or "nothing")
## Waiting on user    (decisions needed; or "nothing")
## Next up            (planned steps)
```
