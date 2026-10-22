# AGENTS.md - Instructions for coding agents

Every coding agent working in this repo follows these rules, whatever tool it is. Humans follow them too.

## Project context

Team Cyclops, ML Challenge Hack4Hack 2026 Round 1 (Planet Hunters). Predict one of 8 labels for each of 7,000 test stars from flux and centroid time series. Metric: macro-F1. Deadline 2026-10-23 10:00 AM IST. Read `docs/PROBLEM_STATEMENT.md` first.

Stack: Python, numpy, pandas, scipy, scikit-learn, LightGBM, astropy, PyTorch (CNN on Kaggle only).

Owner: Jey Pranav owns every issue. Goutam Adityan is on the team but not available for this round.

## Git workflow

- Confirm the branch before starting: `git branch --show-current`
- Work only on the branch named in the task; never switch branches mid-task
- Branch names: `type/issue-number-short-description` (e.g. `feat/4-transit-features`)
- Commit format: `type: short description`, under 72 characters; types: feat, fix, chore, docs, refactor, test
- Stage only files relevant to the task; never `git add .` or `git add -A`
- No co-author trailers and no tool attribution in commits, PRs, issues, comments or files
- Never push to `main`, never merge PRs, never force-push a shared branch
- Never close or edit issues, or comment on PRs, on someone's behalf without their confirmation
- Never create or modify `.github/workflows/` without explicit instruction
- When done, list every file modified or created

## Competition rules that bind the code

- Train only on the provided data. No outside labelled data. No hand-labelling the test set. No pseudo-labelling.
- Using unlabeled test flux for per-CCD trend estimation is allowed; nothing derived from test labels exists.
- Code must reproduce our best upload exactly: fixed seeds (42), one script per submission, every upload logged in `docs/SUBMISSIONS.md` with its commit hash.
- Train and test come from different sectors (momentum dumps every ~150 cadences in train, ~120 in test). Never hard-code cadence indices of dumps; read `quality & 1` from the cadences file.

## File scope

- Only modify files directly related to the task
- Shared contract - flag before changing: `folds.csv`, `src/ph/data.py`, the `oof/` CSV format, `scripts/make_submission.py`
- Never delete files without explicit instruction
- Never commit `data/` or any `.npy`

## Code conventions

- Type hints on public functions; no commented-out code; no leftover debug prints
- Feature functions take a 1D detrended relative-flux array (NaN = gap) and return a dict of floats
- Notebooks: commit with outputs stripped (`pipx run nbstripout notebooks/*.ipynb`); never put tokens in a cell
- Never hardcode secrets; never read or output `.env` contents
