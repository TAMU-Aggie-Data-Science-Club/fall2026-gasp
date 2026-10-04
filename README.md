# G.A.S.P. - Gas Analysis and Storage Prediction

*ADSC Project · Fall 2026*

## Overview

Natural gas is injected into underground storage facilities through the summer and withdrawn through the winter. Every Thursday at 10:30 ET, the EIA publishes how much moved in or out during the week ending the prior Friday. That number is watched closely and prices move on how far it is from analyst consensus.

G.A.S.P. forecasts that number before it is published, three different ways, and measures which approach wins.

## Objective

Predict the weekly change in US working gas in storage (Bcf) ahead of each EIA release, with calibrated uncertainty bands, and benchmark the forecast against naive and seasonal baselines.

Three approaches:
1. **Seasonal Time-series model (SARIMA / SARIMAX)** — predict storage from its own history. Storage is overwhelmingly seasonal.
2. **Gradient Boosting (LightGBM / XGBoost)** — predict from weather, production, and flow features. Statistical pattern matching, no physical structure.
3. **Supply & Demand Balance** — forecast each component of the gas balance separately, then sum the identity:

```
production + imports − (res/comm + power burn + industrial + LNG exports + Mexico exports)
    = change in storage
```


## Tech stack

- **Data processing:** Python, Pandas, SQLite
- **Modeling / ML:** XGBoost, LightGBM, classical time-series (SARIMA, SARIMAX)
- **Visualization:** Plotly, Matplotlib, Streamlit
- **Data sources:** EIA API v2 (storage, production, consumption, LNG exports), NOAA (degree days)

See [`DATA.md`](DATA.md) for concrete data sources and how to access them.

## Scope (v1)

Build:

1. **Ingestion** — EIA API v2 and NOAA into SQLite, on a schedule, with a `retrieved_at` timestamp on every row so we can reconstruct what was knowable on any past date,
2. **Feature engineering** — population-weighted heating and cooling degree days, production, LNG feedgas, lags, week-of-year,
3. **Baselines** — 5-year seasonal average, last-value, SARIMA,
4. **Models** — LightGBM, then the component-wise S&D balance,
5. **Validation** — walk-forward over ~10 years of releases against the 5-year seasonal average and naive baselines,
6. **Dashboard** — current forecast, uncertainty band, component breakdown, and historical accuracy.

**Stretch goal:** benchmark against the analyst consensus if a usable history can be assembled from free trade-press coverage.

**Out of scope for v1:** regional (rather than national) storage, storage asset valuation, futures spread and derivatives pricing, LNG cargo-level modeling from vessel tracking.


See [`DELIVERABLES.md`](DELIVERABLES.md) for the suggested deliverable breakdown and rough timeline.

## Repository map

| File / folder | Purpose |
|---|---|
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | **Start here.** How the team runs the project on GitHub — PM vs. member roles, the issue → PR → `main` flow, branching, worktrees, reviews. |
| [`DELIVERABLES.md`](DELIVERABLES.md) | Suggested deliverables and rough timeline. A living plan, not a contract. |
| [`DATA.md`](DATA.md) | Suggested data sources, how to access them, and the source register. |
| [`data/`](data/) | Local working folder for datasets. **Git-ignored** — data is never committed. |
| [`AGENTS.md`](AGENTS.md) | Machine-facing workflow rules for AI coding agents. |
| [`CODEOWNERS`](CODEOWNERS) | **Team roster + review policy.** PMs, members, and the code-owner rule for PRs into `main`. |


## Team

The current PMs and members for this project are listed in [`CODEOWNERS`](CODEOWNERS). PMs listed there are the code owners for PRs into `main`.
## Notes for PMs

This README, [`DELIVERABLES.md`](DELIVERABLES.md), and [`DATA.md`](DATA.md) are **suggestions**, not commitments. Rewrite them as the team scopes the real project.

## Notes for members

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before touching code. Then pick up an issue from the board.
