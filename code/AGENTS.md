<!-- Claude Code users: copy this file to CLAUDE.md, or put @code/AGENTS.md in your CLAUDE.md. -->

# Instructions for AI agents

This repository is a submission to the Protostar Challenge: Climate Risk Intelligence. Follow these rules in every task.

## Data rule (never break it)

- Forecasts are made as of 2018-12-31. Weather data used as a forecast input must be dated on or before 2018-12-31.
- Do not download, query or use weather observations dated 2019-01-01 or later, from any source (e.g., NOAA, Meteostat, Open-Meteo, ERA5, weather websites), nor 2019–2025 data driven by the weather (e.g., streamflow, air quality, energy demand).
- Set an end date of 2018-12-31 or earlier on every weather download.
- Never write `predicted_value` from memory or from a web search: every value must come from the code's model.
- If a task seems to need weather data after 2018, stop and ask a team member.
- Data that does not respond to the weather (e.g., census, land cover, elevation) may come from any year; record its source in `data/README.md`.

## Working rules

- Training data: `CMU-OSPO/protostar-climate-train-to-2018` (Hugging Face).
- Code goes in `code/`, curated data in `data/` (described in `data/README.md`).
- After changing `submission.csv`, run `python code/check_submission.py`.
- The code must reproduce `submission.csv` without an agent (scripts or notebooks that run top to bottom).
- Never write tokens or credentials into any file.
- Summarize what you did in the README's "AI Tools and Agents" section.

## Team notes

- Use the venv: `python -m venv .venv && .venv/bin/pip install -r code/requirements.txt`. `raw/` and `.venv/` are not part of the submission.
- `climate_lib.load_panel()` raises if any input is dated after 2018: never weaken or remove that check.
- Choose settings only with `code/backtest.py` (origins 1997–2004 for tuning, 2005–2011 to check). Never score variants against 2019–2025 observations.
- A new model must use only years <= `origin`. In `learned.py`, training origins must satisfy origin + 7 <= forecast origin.
- Never pick a model by its live-leaderboard score (that is scoring on 2019–2021 observations).
- After changing the model, rerun `python code/forecast.py`, `python code/check_submission.py`, `python code/check_leakage.py` (must pass), `python code/export.py` and `python code/policy_analysis.py`, and update the numbers in the README.
